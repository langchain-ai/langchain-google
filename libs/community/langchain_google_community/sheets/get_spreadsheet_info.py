"""Tool for retrieving Google Sheets metadata and structure information."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import SheetsBaseTool
from .utils import validate_spreadsheet_id


class GetSpreadsheetInfoSchema(BaseModel):
    """Input schema for `SheetsGetSpreadsheetInfoTool`."""

    spreadsheet_id: str = Field(
        ...,
        description="The ID of the Google Spreadsheet to get information about. "
        "Can be extracted from URL or provided directly.",
    )

    include_grid_data: bool = Field(
        default=False,
        description=(
            "Whether to include detailed grid data (cell properties, formatting). "
            "Note: This can significantly increase response size."
        ),
    )

    include_formatting: bool = Field(
        default=False,
        description="Whether to include cell formatting information when "
        "include_grid_data is True.",
    )

    include_validation: bool = Field(
        default=False,
        description="Whether to include data validation rules when "
        "include_grid_data is True.",
    )

    ranges: Optional[List[str]] = Field(
        default=None,
        description="Specific ranges to get information about. "
        "If None, gets info for all sheets.",
    )

    fields: Optional[str] = Field(
        default=None,
        description="Specific fields to return (e.g., "
        "'sheets.properties.title,sheets.properties.sheetId'). "
        "If None, returns all available fields.",
    )


class SheetsGetSpreadsheetInfoTool(SheetsBaseTool):
    """Tool for retrieving Google Sheets metadata and structure information.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Retrieves spreadsheet properties, sheet details, named ranges, and organizational
    structure. Essential for understanding spreadsheet contents before reading data.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        title (str): Spreadsheet title.
        locale (str): Spreadsheet locale (e.g., 'en_US').
        time_zone (str): Spreadsheet timezone (e.g., 'America/New_York').
        auto_recalc (str): Auto-recalculation setting.
        default_format (dict): Default cell format.
        sheets (list): List of sheet information with properties.
        named_ranges (list): List of named ranges with locations.
        developer_metadata (list): Developer metadata entries.
        grid_data (list): Detailed cell data (when `include_grid_data=True`).

    ??? example "Basic Usage"

        Get basic spreadsheet information:

        ```python
        from langchain_google_community.sheets import SheetsGetSpreadsheetInfoTool

        tool = SheetsGetSpreadsheetInfoTool(api_key="your_api_key")
        result = tool.run(
            {"spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"}
        )
        print(f"Title: {result['title']}")
        print(f"Sheets: {[s['title'] for s in result['sheets']]}")
        ```

    ??? example "With Specific Fields"

        Get only specific fields to reduce response size:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "fields": "properties.title,sheets.properties",
            }
        )
        ```

    ??? example "Include Grid Data"

        Get detailed cell data and formatting:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "include_grid_data": True,
                "include_formatting": True,
                "ranges": ["Sheet1!A1:D10"],
            }
        )
        ```

    Raises:
        ValueError: If spreadsheet_id is invalid.
        Exception: For API errors or connection issues.
    """

    name: str = "sheets_get_spreadsheet_info"

    description: str = (
        "Retrieve comprehensive metadata information from Google Sheets including "
        "spreadsheet properties, sheet details, named ranges, and organizational "
        "structure. Essential for understanding spreadsheet contents before reading "
        "data and exploring spreadsheet structure."
    )

    args_schema: Type[GetSpreadsheetInfoSchema] = GetSpreadsheetInfoSchema

    def _run(
        self,
        spreadsheet_id: str,
        include_grid_data: bool = False,
        include_formatting: bool = False,
        include_validation: bool = False,
        ranges: Optional[List[str]] = None,
        fields: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Get spreadsheet metadata and structure information.

        Args:
            spreadsheet_id: ID of the spreadsheet to retrieve information about.
            include_grid_data: Whether to include detailed grid data.
            include_formatting: Whether to include cell formatting.
            include_validation: Whether to include data validation rules.
            ranges: Specific ranges to get information about.
            fields: Specific fields to return (reduces response size).
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            title (str): Spreadsheet title.
            locale (str): Spreadsheet locale.
            time_zone (str): Spreadsheet timezone.
            auto_recalc (str): Auto-recalculation setting.
            default_format (dict): Default cell format.
            sheets (list): List of sheet information with properties.
            named_ranges (list): List of named ranges with locations.
            developer_metadata (list): Developer metadata entries.
            grid_data (list): Detailed cell data (when `include_grid_data=True`).

        Raises:
            ValueError: If `spreadsheet_id` is invalid.
            Exception: For API errors or connection issues.
        """
        try:
            # Validate spreadsheet ID
            validated_spreadsheet_id = validate_spreadsheet_id(spreadsheet_id)

            # Get the service (handles both OAuth2 and API key auth)
            service = self._get_service()

            # Build the request parameters
            request_params = {
                "spreadsheetId": validated_spreadsheet_id,
                "includeGridData": include_grid_data,
            }

            if ranges:
                request_params["ranges"] = ranges
            if fields:
                request_params["fields"] = fields

            # Get spreadsheet information
            response = service.spreadsheets().get(**request_params).execute()

            # Process the response
            processed_info = self._process_spreadsheet_info(
                response, include_formatting, include_validation
            )

            # Add success field
            processed_info["success"] = True

            return processed_info

        except Exception as error:
            raise Exception(f"Error getting spreadsheet info: {error}") from error

    def _process_spreadsheet_info(
        self,
        response: dict,
        include_formatting: bool = False,
        include_validation: bool = False,
    ) -> dict:
        """Process the raw spreadsheet info response."""
        # Extract basic spreadsheet info
        spreadsheet_info = {
            "spreadsheet_id": response.get("spreadsheetId"),
            "title": response.get("properties", {}).get("title"),
            "locale": response.get("properties", {}).get("locale"),
            "time_zone": response.get("properties", {}).get("timeZone"),
            "auto_recalc": response.get("properties", {}).get("autoRecalc"),
            "default_format": response.get("properties", {}).get("defaultFormat", {}),
            "sheets": [],
            "named_ranges": [],
            "developer_metadata": [],
        }

        # Process sheets information
        sheets = response.get("sheets", [])
        for sheet in sheets:
            sheet_info = {
                "sheet_id": sheet.get("properties", {}).get("sheetId"),
                "title": sheet.get("properties", {}).get("title"),
                "sheet_type": sheet.get("properties", {}).get("sheetType"),
                "grid_properties": sheet.get("properties", {}).get(
                    "gridProperties", {}
                ),
                "tab_color": sheet.get("properties", {}).get("tabColor", {}),
                "hidden": sheet.get("properties", {}).get("hidden", False),
                "right_to_left": sheet.get("properties", {}).get("rightToLeft", False),
            }

            # Add grid data if available
            if "data" in sheet:
                sheet_info["grid_data"] = self._process_grid_data(sheet["data"])

            # Add formatting info if requested
            if include_formatting and "data" in sheet:
                sheet_info["formatting"] = self._extract_formatting(sheet["data"])

            # Add validation info if requested
            if include_validation and "data" in sheet:
                sheet_info["validation"] = self._extract_validation(sheet["data"])

            spreadsheet_info["sheets"].append(sheet_info)

        # Process named ranges
        named_ranges = response.get("namedRanges", [])
        for named_range in named_ranges:
            range_info = {
                "name": named_range.get("name"),
                "range": named_range.get("range"),
                "named_range_id": named_range.get("namedRangeId"),
            }
            spreadsheet_info["named_ranges"].append(range_info)

        # Process developer metadata
        developer_metadata = response.get("developerMetadata", [])
        for metadata in developer_metadata:
            metadata_info = {
                "metadata_id": metadata.get("metadataId"),
                "key": metadata.get("key"),
                "value": metadata.get("value"),
                "visibility": metadata.get("visibility"),
                "location": metadata.get("location", {}),
            }
            spreadsheet_info["developer_metadata"].append(metadata_info)

        return spreadsheet_info

    def _process_grid_data(self, grid_data: List[dict]) -> List[List[str]]:
        """Process ALL grid data segments using simplified patterns.

        Now processes all GridData segments to prevent data loss when the API
        returns multiple segments.
        """
        if not grid_data:
            return []

        # Process ALL grids, not just the first
        result: List[List[str]] = []
        for grid in grid_data:
            for row_data in grid.get("rowData", []) or []:
                row_values: List[str] = []
                for cell_data in row_data.get("values", []) or []:
                    # Use the safe extraction pattern
                    value = self._safe_get_cell_value(cell_data)
                    row_values.append(value)
                result.append(row_values)

        return result

    def _extract_formatting(self, grid_data: List[dict]) -> List[List[dict]]:
        """Extract cell formatting information from ALL grid segments."""
        if not grid_data:
            return []

        # Process ALL grids, not just the first
        formatting_info: List[List[dict]] = []
        for grid in grid_data:
            for row_data in grid.get("rowData", []) or []:
                row_formatting: List[dict] = []
                for cell_data in row_data.get("values", []) or []:
                    cell_formatting = {
                        "user_entered_format": cell_data.get("userEnteredFormat", {}),
                        "effective_format": cell_data.get("effectiveFormat", {}),
                    }
                    row_formatting.append(cell_formatting)
                formatting_info.append(row_formatting)

        return formatting_info

    def _extract_validation(self, grid_data: List[dict]) -> List[List[dict]]:
        """Extract data validation rules from ALL grid segments."""
        if not grid_data:
            return []

        # Process ALL grids, not just the first
        validation_info: List[List[dict]] = []
        for grid in grid_data:
            for row_data in grid.get("rowData", []) or []:
                row_validation: List[dict] = []
                for cell_data in row_data.get("values", []) or []:
                    validation_rule = cell_data.get("dataValidation", {})
                    if validation_rule:
                        row_validation.append(validation_rule)
                    else:
                        row_validation.append({})
                validation_info.append(row_validation)

        return validation_info
