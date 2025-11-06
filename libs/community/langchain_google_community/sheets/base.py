"""Base class for Google Sheets tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.tools import BaseTool
from pydantic import Field

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


class SheetsBaseTool(BaseTool):  # type: ignore[override]
    """Base class for Google Sheets tools.

    Authentication:
    - `api_resource`: OAuth2 credentials for full access (read/write private sheets)
    - `api_key`: API key for read-only access (public sheets only)

    !!! note

        Write operations require OAuth2 credentials.
    """

    api_resource: Optional[Resource] = Field(
        default=None,
        description="Google Sheets API resource, OAuth2 credentials for full access",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="Google API key for read-only access to public spreadsheets",
    )

    def _get_service(self) -> Resource:
        """Get the appropriate Google Sheets service based on available credentials.

        Returns:
            Resource: Google Sheets API service

        Raises:
            ValueError: If neither `api_resource` nor `api_key` is provided
        """
        if self.api_resource:
            return self.api_resource  # OAuth2 - full access
        elif self.api_key:
            from langchain_google_community.sheets.utils import (
                build_sheets_service_with_api_key,
            )

            return build_sheets_service_with_api_key(
                self.api_key
            )  # API key - read-only
        else:
            # Try to create OAuth2 service as fallback
            from langchain_google_community.sheets.utils import build_sheets_service

            return build_sheets_service()

    def _check_write_permissions(self) -> None:
        """Check if the current authentication method supports write operations.

        Raises:
            ValueError: If using API key for write operations
        """
        if self.api_key and not self.api_resource:
            raise ValueError(
                "Write operations require OAuth2 credentials, not API key. "
                "Please provide api_resource for write access to private spreadsheets."
            )

    @classmethod
    def from_api_resource(cls, api_resource: Resource) -> "SheetsBaseTool":
        """Create a tool from an API resource.

        Args:
            api_resource: The API resource to use.

        Returns:
            A tool with OAuth2 credentials for full access.
        """
        return cls(api_resource=api_resource)  # type: ignore[call-arg]

    @classmethod
    def from_api_key(cls, api_key: str) -> "SheetsBaseTool":
        """Create a tool from an API key.

        Args:
            api_key: The API key to use.

        Returns:
            A tool with API key for read-only access.
        """
        return cls(api_key=api_key)  # type: ignore[call-arg]

    def _safe_get_cell_value(self, cell_data: dict) -> str:
        """Safely extract cell value with proper fallback hierarchy.

        Args:
            cell_data: Cell data dictionary from Google Sheets API

        Returns:
            The cell value as a string.
        """
        if cell_data.get("formattedValue"):
            return cell_data["formattedValue"]
        elif cell_data.get("effectiveValue", {}).get("stringValue"):
            return str(cell_data["effectiveValue"]["stringValue"])
        elif cell_data.get("effectiveValue", {}).get("numberValue") is not None:
            return str(cell_data["effectiveValue"]["numberValue"])
        elif cell_data.get("effectiveValue", {}).get("boolValue") is not None:
            return str(cell_data["effectiveValue"]["boolValue"])
        elif cell_data.get("userEnteredValue", {}).get("stringValue"):
            return str(cell_data["userEnteredValue"]["stringValue"])
        elif cell_data.get("userEnteredValue", {}).get("numberValue") is not None:
            return str(cell_data["userEnteredValue"]["numberValue"])
        elif cell_data.get("userEnteredValue", {}).get("boolValue") is not None:
            return str(cell_data["userEnteredValue"]["boolValue"])
        else:
            return ""

    def _convert_to_dict_list(self, items: list) -> list:
        """Convert a list of items to dictionaries.

        Handles both Pydantic models and dicts. Useful for converting
        user-provided schemas (which may be Pydantic models or plain dicts)
        into the dict format required by Google Sheets API.

        Args:
            items: List of items that may be Pydantic models or dictionaries

        Returns:
            List of dictionaries with all Pydantic models converted.
        """
        result = []
        for item in items:
            if hasattr(item, "model_dump"):
                # It's a Pydantic model, convert to dict
                result.append(item.model_dump())
            else:
                # It's already a dict
                result.append(item)
        return result
