"""Chain that calls Google Places API."""

import logging
from typing import Any, Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class GooglePlacesAPIWrapper(BaseModel):
    """Wrapper around Google Places API.

    Searches for places using Google Maps Platform. Returns detailed information
    including addresses, phone numbers, and websites.

    !!! note "Installation"

        Requires additional dependencies:

        ```bash
        pip install langchain-google-community[places]
        ```

    !!! note "Setup Required"

        Set `GPLACES_API_KEY` environment variable or pass `gplaces_api_key`
        parameter with your Google Maps Platform API key.
    """

    gplaces_api_key: Optional[str] = None
    """Google Maps Platform API key."""

    google_map_client: Any = None

    top_k_results: Optional[int] = None
    """Maximum number of results to return."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that API key is in your environment variable."""
        gplaces_api_key = get_from_dict_or_env(
            values, "gplaces_api_key", "GPLACES_API_KEY"
        )
        values["gplaces_api_key"] = gplaces_api_key
        try:
            import googlemaps  # type: ignore[import]

            values["google_map_client"] = googlemaps.Client(gplaces_api_key)
        except ImportError:
            raise ImportError(
                "Could not import googlemaps python package. "
                "Please, install places dependency group: "
                "`pip install langchain-google-community[places]`"
            )
        return values

    def run(self, query: str) -> str:
        """Run Places search and get k number of places that exists that match."""
        search_results = self.google_map_client.places(query)["results"]
        num_to_return = len(search_results)

        places = []

        if num_to_return == 0:
            return "Google Places did not find any places that match the description"

        num_to_return = (
            num_to_return
            if self.top_k_results is None
            else min(num_to_return, self.top_k_results)
        )

        for i in range(num_to_return):
            result = search_results[i]
            details = self.fetch_place_details(result["place_id"])

            if details is not None:
                places.append(details)

        return "\n".join([f"{i + 1}. {item}" for i, item in enumerate(places)])

    def fetch_place_details(self, place_id: str) -> Optional[str]:
        try:
            place_details = self.google_map_client.place(place_id)
            place_details["place_id"] = place_id
            formatted_details = self.format_place_details(place_details)
            return formatted_details
        except Exception as e:
            logging.error(f"An Error occurred while fetching place details: {e}")
            return None

    def format_place_details(self, place_details: Dict[str, Any]) -> Optional[str]:
        try:
            name = place_details.get("result", {}).get("name", "Unknown")
            address = place_details.get("result", {}).get(
                "formatted_address", "Unknown"
            )
            phone_number = place_details.get("result", {}).get(
                "formatted_phone_number", "Unknown"
            )
            website = place_details.get("result", {}).get("website", "Unknown")
            place_id = place_details.get("result", {}).get("place_id", "Unknown")

            formatted_details = (
                f"{name}\nAddress: {address}\n"
                f"Google place ID: {place_id}\n"
                f"Phone: {phone_number}\nWebsite: {website}\n\n"
            )
            return formatted_details
        except Exception as e:
            logging.error(f"An error occurred while formatting place details: {e}")
            return None


class GooglePlacesSchema(BaseModel):
    """Input schema for `GooglePlacesTool`."""

    query: str = Field(...)
    """Search query for Google Maps"""


class GooglePlacesTool(BaseTool):
    """Tool that queries the Google Places API.

    Inherits from [`BaseTool`][langchain_core.tools.BaseTool].

    Validates and discovers addresses from ambiguous text using Google Maps Platform.
    """

    name: str = "google_places"

    description: str = (
        "A wrapper around Google Places. "
        "Useful for when you need to validate or "
        "discover addresses from ambiguous text. "
        "Input should be a search query."
    )

    api_wrapper: GooglePlacesAPIWrapper = Field(default_factory=GooglePlacesAPIWrapper)  # type: ignore[arg-type]

    args_schema: Type[BaseModel] = GooglePlacesSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search for places matching the query.

        Args:
            query: Search query for Google Maps.
            run_manager: Optional callback manager.

        Returns:
            Formatted string with place details for each result.
        """
        return self.api_wrapper.run(query)
