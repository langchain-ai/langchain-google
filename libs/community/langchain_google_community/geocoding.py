"""Google Maps Geocoding API wrapper for LangChain.

Note: The googlemaps package doesn't provide type stubs, but we provide complete
type hints for our wrapper implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _import_googlemaps():
    """Import the googlemaps library."""
    try:
        import googlemaps  # type: ignore

        return googlemaps
    except ImportError:
        raise ImportError(
            "googlemaps package not found, please install it with "
            "`pip install googlemaps`"
        )


class GoogleGeocodingAPIWrapper:
    """Wrapper for Google Maps Geocoding API.

    Docs: https://developers.google.com/maps/documentation/geocoding/overview

    To use this wrapper, you need a Google Maps API key.
    You can get one here: https://developers.google.com/maps/documentation/geocoding/get-api-key
    """

    def __init__(self, api_key: str):
        """Initialize the wrapper.

        Args:
            api_key: The API key to use.
        """
        googlemaps = _import_googlemaps()
        self.api_key = api_key
        self.client = googlemaps.Client(key=api_key)

    def geocode(
        self,
        address: str,
        components: Optional[Dict[str, str]] = None,
        bounds: Optional[Dict[str, Dict[str, float]]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Geocode an address.

        Args:
            address: The address to geocode.
            components: A components filter with elements separated by a pipe (|).
            bounds: The bounding box of the viewport within which to bias geocode results.
            region: The region code, specified as a ccTLD two-character value.
            language: The language in which to return results.

        Returns:
            A string containing the formatted address and coordinates.
        """
        try:
            results = self.client.geocode(
                address=address,
                components=components,
                bounds=bounds,
                region=region,
                language=language,
            )

            if not results:
                return "No results found for the given address."

            result = results[0]
            formatted_address = result["formatted_address"]
            location = result["geometry"]["location"]
            lat = location["lat"]
            lng = location["lng"]

            return f"Address: {formatted_address}\n" f"Coordinates: {lat}, {lng}"
        except Exception as e:
            raise ValueError(f"Geocoding request failed: {str(e)}")


class GeocodingInput(BaseModel):
    """Input for GoogleGeocodingTool."""

    address: str = Field(..., description="The address to geocode")
    components: Optional[Dict[str, str]] = Field(
        None,
        description="A components filter with elements separated by a pipe (|). "
        "Format: country:US|locality:santa cruz",
    )
    bounds: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="The bounding box to bias results. "
        "Format: {'northeast': {'lat': float, 'lng': float}, "
        "'southwest': {'lat': float, 'lng': float}}",
    )
    region: Optional[str] = Field(
        None,
        description="The region code, specified as a ccTLD two-character value",
    )
    language: Optional[str] = Field(
        None,
        description="The language in which to return results",
    )


class GoogleGeocodingTool(BaseTool):
    """Tool that queries the Google Maps Geocoding API."""

    name: str = "google_geocoding"
    description: str = (
        "A wrapper around Google Maps Geocoding API. "
        "Useful for converting addresses into geographic coordinates (latitude/longitude). "
        "Input should be a physical address or place name."
    )
    api_wrapper: GoogleGeocodingAPIWrapper
    args_schema: Type[BaseModel] = GeocodingInput

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initialize the tool.

        Args:
            api_key: The API key to use.
            **kwargs: Additional arguments to pass to the tool.
        """
        super().__init__(
            api_wrapper=GoogleGeocodingAPIWrapper(api_key=api_key), **kwargs
        )

    def _run(
        self,
        address: str,
        components: Optional[Dict[str, str]] = None,
        bounds: Optional[Dict[str, Dict[str, float]]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool.

        Args:
            address: The address to geocode.
            components: A components filter.
            bounds: The bounding box to bias results.
            region: The region code.
            language: The language for results.
            run_manager: The callback manager.

        Returns:
            The geocoding results as a string.
        """
        return self.api_wrapper.geocode(
            address=address,
            components=components,
            bounds=bounds,
            region=region,
            language=language,
        )
