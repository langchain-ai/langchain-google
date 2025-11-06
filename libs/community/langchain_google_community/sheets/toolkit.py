from typing import TYPE_CHECKING, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_google_community.sheets.create_spreadsheet_tool import (
    SheetsCreateSpreadsheetTool,
)
from langchain_google_community.sheets.get_spreadsheet_info import (
    SheetsGetSpreadsheetInfoTool,
)
from langchain_google_community.sheets.read_sheet_tools import (
    SheetsBatchReadDataTool,
    SheetsFilteredReadDataTool,
    SheetsReadDataTool,
)
from langchain_google_community.sheets.write_sheet_tools import (
    SheetsAppendValuesTool,
    SheetsBatchUpdateValuesTool,
    SheetsClearValuesTool,
    SheetsUpdateValuesTool,
)

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


class SheetsToolkit(BaseToolkit):
    """Toolkit for interacting with Google Sheets.

    Inherits from [`BaseToolkit`][langchain_core.tools.base.BaseToolkit].

    Provides comprehensive Google Sheets integration with read and write capabilities.

    !!! warning "Security Note"

        This toolkit contains tools that can read and write data to Google Sheets.
        Ensure proper authentication and access controls.

    !!! info "Authentication Requirements"

        - **Read operations**: Require only API key (for public spreadsheets)
        - **Write operations**: Require OAuth2 credentials (`api_resource`)
    """

    api_resource: Resource = Field(default=None)  # type: ignore[assignment]

    api_key: Optional[str] = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit.

        Returns:
            List of tools based on authentication method. API key provides read-only
            tools. OAuth2 provides full read/write tools.
        """
        # If api_key is provided, return read-only tools
        if self.api_key:
            return [
                SheetsReadDataTool(api_key=self.api_key),
                SheetsBatchReadDataTool(api_key=self.api_key),
                SheetsGetSpreadsheetInfoTool(api_key=self.api_key),
                # Note: Write operations and FilteredReadDataTool require OAuth2
            ]

        # Otherwise, use the api_resource (OAuth2) for full read/write access
        return [
            # Read operations
            SheetsReadDataTool(api_resource=self.api_resource),
            SheetsBatchReadDataTool(api_resource=self.api_resource),
            SheetsFilteredReadDataTool(api_resource=self.api_resource),
            SheetsGetSpreadsheetInfoTool(api_resource=self.api_resource),
            # Write operations (OAuth2 only)
            SheetsCreateSpreadsheetTool(api_resource=self.api_resource),
            SheetsUpdateValuesTool(api_resource=self.api_resource),
            SheetsAppendValuesTool(api_resource=self.api_resource),
            SheetsClearValuesTool(api_resource=self.api_resource),
            SheetsBatchUpdateValuesTool(api_resource=self.api_resource),
        ]
