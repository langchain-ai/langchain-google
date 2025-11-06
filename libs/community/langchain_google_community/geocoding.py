"""Wrapper and Tool for the Google Maps Geocoding API."""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"


class GoogleGeocodingAPIWrapper(BaseModel):
    """Wrapper for Google Maps Geocoding API.

    Provides methods for geocoding locations with configurable options for
    address components, geometry, metadata, and navigation points.
    """

    # Required
    google_api_key: SecretStr

    # Configuration
    include_bounds: bool = Field(default=True)
    include_navigation: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    include_address_components: bool = Field(default=True)

    # Default parameters
    language: Optional[str] = Field(default="en")
    region: Optional[str] = Field(default="us")
    max_retries: int = Field(default=2)
    timeout: int = Field(default=30)

    model_config = ConfigDict(extra="forbid")

    def _add_components(self, cleaned: Dict, result: Dict) -> None:
        """Add address components if configured."""
        if not self.include_address_components:
            return

        address_types = [
            "street_number",
            "route",
            "locality",
            "country",
            "postal_code",
        ]

        for component in result.get("address_components", []):
            types = component.get("types", [])
            name = component.get("long_name")
            for type_ in types:
                if type_ in address_types:
                    cleaned["address"][type_] = name
                elif type_ == "administrative_area_level_1":
                    state_name = component.get("short_name")
                    cleaned["address"]["state"] = state_name

    def _add_geometry(self, cleaned: Dict, result: Dict) -> None:
        """Add geometry details if configured."""
        geometry = result.get("geometry", {})
        if self.include_bounds:
            if geometry.get("viewport"):
                cleaned["geometry"]["viewport"] = geometry["viewport"]
            if geometry.get("bounds"):
                cleaned["geometry"]["bounds"] = geometry["bounds"]

    def _add_metadata(self, cleaned: Dict, result: Dict) -> None:
        """Add metadata if configured."""
        if self.include_metadata:
            geometry = result.get("geometry", {})
            cleaned["metadata"] = {
                "place_id": result.get("place_id"),
                "types": result.get("types", []),
                "location_type": (geometry.get("location_type")),
            }

    def _add_navigation(self, cleaned: Dict, result: Dict) -> None:
        """Add navigation points if configured."""
        if self.include_navigation and result.get("navigation_points"):
            cleaned["navigation"] = [
                {
                    "location": point["location"],
                    "restrictions": point.get("restricted_travel_modes", []),
                }
                for point in result.get("navigation_points", [])
            ]

    def clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean and format results."""
        cleaned_results = []

        for result in results:
            if not result:
                continue

            cleaned = {
                "address": {"full": result.get("formatted_address", "")},
                "geometry": {
                    "location": result.get("geometry", {}).get("location", {})
                },
            }

            # Add optional components
            self._add_components(cleaned, result)
            self._add_geometry(cleaned, result)
            self._add_metadata(cleaned, result)
            self._add_navigation(cleaned, result)

            cleaned_results.append(cleaned)

        return cleaned_results

    def results(
        self,
        query: str,
        language: Optional[str] = None,
        region: Optional[str] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """Process geocoding request and return results.

        Handles both single and batch geocoding requests with detailed location data.

        Args:
            query: Location(s) to geocode (e.g., `'Eiffel Tower'` or
                `'Times Square, Central Park'`).
            language: Language code for results (e.g., `'en'`, `'fr'`, `'ja'`).
            region: Region bias (e.g., `'us'`, `'fr'`, `'jp'`).
            max_results: Maximum number of results to return.

        Returns:
            status (str): Request status (`'OK'` or error status).
            total_results (int): Number of locations found.
            results (list): Location data with address, geometry, metadata, and
                navigation.
            query_info (dict): Query metadata (original query, language, region).
        """
        try:
            if not query.strip():
                return {
                    "status": "ERROR",
                    "message": "Empty query provided",
                    "results": [],
                }

            # Handle batch queries
            queries = [q.strip() for q in query.split(",") if q.strip()]
            if len(queries) > max_results:
                queries = queries[:max_results]

            raw_results = self.raw_results(
                query=query, language=language, region=region
            )

            if raw_results.get("status") != "OK":
                return {
                    "status": raw_results.get("status", "ERROR"),
                    "message": raw_results.get("error_message", "No results found"),
                    "results": [],
                }

            return {
                "status": "OK",
                "total_results": len(raw_results.get("results", [])),
                "results": self.clean_results(raw_results.get("results", [])),
                "query_info": {
                    "original_query": query,
                    "language": language or self.language,
                    "region": region or self.region,
                },
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "message": str(e),
                "results": [],
                "query_info": {
                    "original_query": query,
                    "language": language or self.language,
                    "region": region or self.region,
                },
            }

    def batch_geocode(
        self,
        locations: List[str],
        language: Optional[str] = None,
        region: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Process multiple locations in batch.

        Args:
            locations: List of locations (e.g., `["Eiffel Tower", "Times Square"]`).
            language: Language code for results.
            region: Region bias.
            components: Component filters (e.g., `{"country": "US"}`).

        Returns:
            status (str): Overall batch status.
            total_results (int): Number of successful results.
            results (list): Location data for successful queries.
            errors (list): Errors encountered with query, status, and message.
            query_info (dict): Batch metadata (total, successful, failed counts).
        """
        if not locations:
            return {
                "status": "ERROR",
                "message": "No locations provided",
                "results": [],
            }

        results = []
        errors = []

        for location in locations:
            try:
                result = self.raw_results(
                    query=location,
                    language=language,
                    region=region,
                    components=components,
                )

                if result.get("status") == "OK":
                    results.append(result)
                else:
                    errors.append(
                        {
                            "query": location,
                            "status": result.get("status"),
                            "message": result.get("error_message"),
                        }
                    )
            except Exception as e:
                errors.append({"query": location, "status": "ERROR", "message": str(e)})

        return {
            "status": "OK" if results else "ERROR",
            "total_results": len(results),
            "results": self.clean_results(
                [r.get("results", [])[0] for r in results if r.get("results")]
            ),
            "errors": errors,
            "query_info": {
                "total_queries": len(locations),
                "successful": len(results),
                "failed": len(errors),
                "language": language or self.language,
                "region": region or self.region,
            },
        }

    def raw_results(
        self,
        query: str,
        language: Optional[str] = None,
        region: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Get raw results with improved error handling."""
        try:
            # Input validation
            if not query.strip():
                return {
                    "status": "ERROR",
                    "error_message": "Empty query provided",
                    "results": [],
                }

            # Build parameters
            params = {
                "address": query.strip(),
                "key": self.google_api_key.get_secret_value(),
                "language": language or self.language,
                "region": region or self.region,
            }

            # Add component filtering if provided
            if components:
                params["components"] = "|".join(
                    f"{k}:{v}" for k, v in components.items()
                )

            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        GOOGLE_MAPS_API_URL, params=params, timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                    if data.get("status") == "OK":
                        return data
                    elif attempt == self.max_retries - 1:
                        return {
                            "status": data.get("status"),
                            "error_message": self._get_error_message(
                                data.get("status")
                            ),
                            "results": [],
                        }
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        return {
                            "status": "REQUEST_ERROR",
                            "error_message": f"Request failed: {str(e)}",
                            "results": [],
                        }

        except Exception as e:
            return {
                "status": "ERROR",
                "error_message": f"Processing error: {str(e)}",
                "results": [],
            }

        # Add explicit return for the case when all retries fail
        return {
            "status": "ERROR",
            "error_message": "All retry attempts failed",
            "results": [],
        }

    def _get_error_message(self, status: str) -> str:
        """Get detailed error message based on status code."""
        error_messages = {
            "ZERO_RESULTS": "No results found for this query",
            "OVER_DAILY_LIMIT": "API key quota exceeded",
            "OVER_QUERY_LIMIT": "Query limit exceeded",
            "REQUEST_DENIED": "Request was denied, check API key",
            "INVALID_REQUEST": "Invalid request parameters",
            "MAX_ELEMENTS_EXCEEDED": "Too many locations in request",
            "UNKNOWN_ERROR": "Server error, please try again",
        }
        return error_messages.get(status, f"API Error: {status}")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_MAPS_API_KEY"
        )
        values["google_api_key"] = google_api_key
        return values

    async def geocode_async(
        self,
        query: str,
        language: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Geocode query asynchronously.

        Args:
            query: Location(s) to geocode.
            language: Language code for results.
            region: Region bias.

        Returns:
            status (str): Request status.
            results (list): Geocoding results.
            query_info (dict): Request metadata.
        """
        try:
            params: Dict[str, str] = {
                "address": query.strip(),
                "key": self.google_api_key.get_secret_value(),
                "language": language or self.language or "",
                "region": region or self.region or "",
            }
            timeout_obj = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GOOGLE_MAPS_API_URL, params=params, timeout=timeout_obj
                ) as response:
                    data = await response.json()
                    if data.get("status") == "OK":
                        return self.results(query, language, region)
                    return {
                        "status": data.get("status", "ERROR"),
                        "error_message": data.get("error_message", "Request failed"),
                        "results": [],
                    }

        except Exception as e:
            return {
                "status": "ERROR",
                "error_message": f"Async request failed: {str(e)}",
                "results": [],
            }


class GoogleGeocodeInput(BaseModel):
    """Input schema for `GoogleGeocodingTool`."""

    query: str = Field(description="Locations for query.")


class GoogleGeocodingTool(BaseTool):
    """Tool for geocoding locations using Google Maps Geocoding API.

    Inherits from [`BaseTool`][langchain_core.tools.BaseTool].

    Supports batch location lookups with detailed address and coordinate information.

    """

    name: str = "google_geocode"

    description: str = (
        "A geocoding tool for multiple locations. "
        "Input: comma-separated locations. "
        "Returns: location data."
    )

    args_schema: Type[BaseModel] = GoogleGeocodeInput

    # Configuration
    max_results: int = 5
    include_bounds: bool = Field(default=True)
    include_navigation: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    language: Optional[str] = Field(default="en")
    region: Optional[str] = Field(default=None)

    api_wrapper: GoogleGeocodingAPIWrapper = Field(
        default_factory=lambda: GoogleGeocodingAPIWrapper(
            google_api_key=SecretStr(os.getenv("GOOGLE_MAPS_API_KEY", ""))
        )
    )
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Geocode locations.

        Args:
            query: Comma-separated locations to geocode.
            run_manager: Callback manager.

        Returns:
            results: Location data with coordinates and addresses.
            response: Raw response with status and query info.
        """
        try:
            locations = [loc.strip() for loc in query.split(",") if loc.strip()]
            if len(locations) > self.max_results:
                locations = locations[: self.max_results]

            if len(locations) > 1:
                results = self.api_wrapper.batch_geocode(
                    locations=locations, language=self.language, region=self.region
                )
            else:
                raw_results = self.api_wrapper.raw_results(
                    query=locations[0], language=self.language, region=self.region
                )
                results = {
                    "status": raw_results.get("status"),
                    "total_results": 1,
                    "results": self.api_wrapper.clean_results(
                        raw_results.get("results", [])
                    ),
                    "query_info": {
                        "total_queries": 1,
                        "successful": 1 if raw_results.get("status") == "OK" else 0,
                        "failed": 0 if raw_results.get("status") == "OK" else 1,
                    },
                }

            if results.get("status") != "OK":
                return [], {"error": results.get("error_message", "Geocoding failed")}

            return results["results"], results

        except Exception as e:
            return [], {"error": str(e)}

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Geocode locations asynchronously.

        Args:
            query: Comma-separated locations to geocode.
            run_manager: Async callback manager.

        Returns:
            results: Location data with coordinates and addresses.
            response: Raw response with status and query info.
        """
        try:
            result = await self.api_wrapper.geocode_async(
                query=query, language=self.language, region=self.region
            )
            if result.get("status") != "OK":
                return [], {"error": result.get("error_message", "Geocoding failed")}
            return result.get("results", []), result
        except Exception as e:
            return [], {"error": str(e)}
