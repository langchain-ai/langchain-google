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
    - api_resource: OAuth2 credentials for full access (read/write private sheets)
    - api_key: API key for read-only access (public sheets only)

    Note: Write operations require OAuth2 credentials.
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
            ValueError: If neither api_resource nor api_key is provided
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
