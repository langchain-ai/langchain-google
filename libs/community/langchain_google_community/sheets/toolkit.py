from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_google_community.sheets.get_spreadsheet_info import (
    SheetsGetSpreadsheetInfoTool,
)
from langchain_google_community.sheets.read_sheet_tools import (
    SheetsBatchReadDataTool,
    SheetsFilteredReadDataTool,
    SheetsReadDataTool,
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

    *Security Note*: This toolkit contains tools that can read data from
        Google Sheets. Currently, only read operations are supported.

        For example, this toolkit can be used to read spreadsheet data,
        get spreadsheet metadata, and perform filtered data queries.

        See https://python.langchain.com/docs/security for more information.
    """

    api_resource: Resource = Field(default=None)
    api_key: str | None = Field(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        # If api_key is provided, use it for all tools
        if self.api_key:
            return [
                SheetsReadDataTool(api_key=self.api_key),
                SheetsBatchReadDataTool(api_key=self.api_key),
                SheetsGetSpreadsheetInfoTool(api_key=self.api_key),
                # Note: FilteredReadDataTool requires OAuth2, not API key
            ]

        # Otherwise, use the api_resource (OAuth2)
        return [
            SheetsReadDataTool(api_resource=self.api_resource),
            SheetsBatchReadDataTool(api_resource=self.api_resource),
            SheetsFilteredReadDataTool(api_resource=self.api_resource),
            SheetsGetSpreadsheetInfoTool(api_resource=self.api_resource),
        ]
