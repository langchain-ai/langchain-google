"""Tools for writing data to Google Sheets."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import SheetsBaseTool
from .enums import InsertDataOption, ValueInputOption
from .utils import (
    validate_range_name,
    validate_spreadsheet_id,
)

# ============================================================================
# 1. BASE SCHEMAS
# ============================================================================


class WriteBaseSchema(BaseModel):
    """Base schema for write operations with common fields."""

    spreadsheet_id: str = Field(
        description="The ID of the Google Spreadsheet to write to."
    )

    value_input_option: ValueInputOption = Field(
        default=ValueInputOption.USER_ENTERED,
        description=(
            "How to interpret input values. "
            "'RAW' stores values as-is. "
            "'USER_ENTERED' parses values as if typed by user."
        ),
    )


# ============================================================================
# 2. UPDATE VALUES (Schema + Tool)
# ============================================================================


class UpdateValuesSchema(WriteBaseSchema):
    """Input schema for `SheetsUpdateValuesTool`."""

    range: str = Field(
        description="The A1 notation of the range to update (e.g., 'Sheet1!A1:C3')."
    )

    values: List[List[Any]] = Field(
        description=(
            "2D array of values to write. Each inner array represents a row. "
            "Supports strings, numbers, booleans, and formulas."
        )
    )


class SheetsUpdateValuesTool(SheetsBaseTool):
    """Tool for updating values in a single range of Google Sheets.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Updates cell values in a specified range, overwriting existing data.

    !!! note "Authentication Required"

        Requires OAuth2 authentication. Use `api_resource` parameter with
        authenticated Google Sheets service.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        updated_range (str): The A1 notation of updated range.
        updated_rows (int): Number of rows updated.
        updated_columns (int): Number of columns updated.
        updated_cells (int): Total number of cells updated.

    ???+ example "Basic Usage"

        Update a range with data:

        ```python
        from langchain_google_community.sheets import SheetsUpdateValuesTool

        tool = SheetsUpdateValuesTool(api_resource=service)
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "range": "Sheet1!A1:C3",
                "values": [
                    ["Name", "Age", "City"],
                    ["Alice", "25", "New York"],
                    ["Bob", "30", "San Francisco"],
                ],
            }
        )
        ```

    ??? example "With Formulas"

        Update cells with formulas using `USER_ENTERED`:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "range": "Sheet1!D2:D3",
                "values": [["=SUM(B2:C2)"], ["=SUM(B3:C3)"]],
                "value_input_option": "USER_ENTERED",
            }
        )
        ```

    Raises:
        ValueError: If write permissions unavailable or validation fails.
        Exception: If `spreadsheet_id` is invalid, range is malformed, or API
            errors occur.
    """

    name: str = "sheets_update_values"

    description: str = (
        "Update values in a single range of a Google Spreadsheet. "
        "Overwrites existing data in the specified range."
    )

    args_schema: Type[BaseModel] = UpdateValuesSchema

    def _run(
        self,
        spreadsheet_id: str,
        range: str,
        values: List[List[Any]],
        value_input_option: ValueInputOption = ValueInputOption.USER_ENTERED,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Update values in a Google Spreadsheet range.

        Args:
            spreadsheet_id: ID of the spreadsheet to update.
            range: A1 notation range to update (e.g., `'Sheet1!A1:C3'`).
            values: 2D array of values to write.
            value_input_option: How to interpret input values.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            updated_range (str): The A1 notation of updated range.
            updated_rows (int): Number of rows updated.
            updated_columns (int): Number of columns updated.
            updated_cells (int): Total number of cells updated.

        Raises:
            ValueError: If write permissions unavailable or validation fails.
            Exception: For API errors or connection issues.
        """
        # Check write permissions (requires OAuth2)
        self._check_write_permissions()

        try:
            # Validate inputs
            spreadsheet_id = validate_spreadsheet_id(spreadsheet_id)
            range = validate_range_name(range)
            # value_input_option is already validated by Pydantic (it's an Enum)

            # Get the Google Sheets service
            service = self._get_service()

            # Build the request body
            body = {"values": values}

            # Update the values
            result = (
                service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range,
                    valueInputOption=value_input_option.value,
                    body=body,
                )
                .execute()
            )

            return {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "updated_range": result.get("updatedRange"),
                "updated_rows": result.get("updatedRows", 0),
                "updated_columns": result.get("updatedColumns", 0),
                "updated_cells": result.get("updatedCells", 0),
            }

        except Exception as error:
            raise Exception(f"Error updating sheet data: {error}") from error


# ============================================================================
# 3. APPEND VALUES (Schema + Tool)
# ============================================================================


class AppendValuesSchema(WriteBaseSchema):
    """Input schema for `SheetsAppendValuesTool`."""

    range: str = Field(
        description=(
            "The A1 notation of the table range to append to. "
            "The API will find the last row of data and append below it."
        )
    )

    values: List[List[Any]] = Field(
        description=(
            "2D array of values to append. Each inner array represents a row. "
            "Supports strings, numbers, booleans, and formulas."
        )
    )

    insert_data_option: InsertDataOption = Field(
        default=InsertDataOption.INSERT_ROWS,
        description=(
            "How to handle existing data. "
            "'OVERWRITE' overwrites data after table. "
            "'INSERT_ROWS' inserts new rows for data."
        ),
    )


class SheetsAppendValuesTool(SheetsBaseTool):
    """Tool for appending values to a Google Spreadsheet table.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Appends data to the end of a table, automatically finding the last row with
    data.

    !!! note "Authentication Required"

        Requires OAuth2 authentication. Use `api_resource` parameter with
        authenticated Google Sheets service.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        table_range (str): The range of the entire table.
        updated_range (str): The specific range where data was appended.
        updated_rows (int): Number of rows appended.
        updated_columns (int): Number of columns appended.
        updated_cells (int): Total number of cells updated.

    ???+ example "Basic Usage"

        Append new records to a table:

        ```python
        from langchain_google_community.sheets import SheetsAppendValuesTool

        tool = SheetsAppendValuesTool(api_resource=service)
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "range": "Sheet1!A1:D100",
                "values": [
                    ["Eve", "27", "Seattle", "91"],
                    ["Frank", "32", "Denver", "85"],
                ],
            }
        )
        ```

    ??? example "With Insert Rows Option"

        Insert rows instead of overwriting:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "range": "Sheet1!A1:D100",
                "values": [["New", "Record", "Data", "Here"]],
                "insert_data_option": "INSERT_ROWS",
            }
        )
        ```

    Raises:
        ValueError: If write permissions unavailable or validation fails.
        Exception: If `spreadsheet_id` is invalid, range is malformed, or API
            errors occur.
    """

    name: str = "sheets_append_values"

    description: str = (
        "Append values to a table in a Google Spreadsheet. "
        "Automatically finds the last row and appends data below it."
    )

    args_schema: Type[BaseModel] = AppendValuesSchema

    def _run(
        self,
        spreadsheet_id: str,
        range: str,
        values: List[List[Any]],
        value_input_option: ValueInputOption = ValueInputOption.USER_ENTERED,
        insert_data_option: InsertDataOption = InsertDataOption.INSERT_ROWS,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Append values to a Google Spreadsheet table.

        Args:
            spreadsheet_id: ID of the spreadsheet.
            range: A1 notation table range to append to.
            values: 2D array of values to append.
            value_input_option: How to interpret input values.
            insert_data_option: How to handle existing data.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            table_range (str): The range of the entire table.
            updated_range (str): The specific range where data was appended.
            updated_rows (int): Number of rows appended.
            updated_columns (int): Number of columns appended.
            updated_cells (int): Total number of cells updated.

        Raises:
            ValueError: If write permissions unavailable or validation fails.
            Exception: For API errors or connection issues.
        """
        # Check write permissions (requires OAuth2)
        self._check_write_permissions()

        try:
            # Validate inputs
            spreadsheet_id = validate_spreadsheet_id(spreadsheet_id)
            range = validate_range_name(range)
            # Enum parameters are already validated by Pydantic

            # Get the Google Sheets service
            service = self._get_service()

            # Build the request body
            body = {"values": values}

            # Append the values
            result = (
                service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range,
                    valueInputOption=value_input_option.value,
                    insertDataOption=insert_data_option.value,
                    body=body,
                )
                .execute()
            )

            updates = result.get("updates", {})

            return {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "table_range": result.get("tableRange"),
                "updated_range": updates.get("updatedRange"),
                "updated_rows": updates.get("updatedRows", 0),
                "updated_columns": updates.get("updatedColumns", 0),
                "updated_cells": updates.get("updatedCells", 0),
            }

        except Exception as error:
            raise Exception(f"Error appending sheet data: {error}") from error


# ============================================================================
# 4. CLEAR VALUES (Schema + Tool)
# ============================================================================


class ClearValuesSchema(BaseModel):
    """Input schema for `SheetsClearValuesTool`."""

    spreadsheet_id: str = Field(description="The ID of the Google Spreadsheet.")

    range: str = Field(
        description=(
            "The A1 notation of the range to clear (e.g., 'Sheet1!A1:Z100'). "
            "Only values are cleared; formatting remains."
        )
    )


class SheetsClearValuesTool(SheetsBaseTool):
    """Tool for clearing values from a Google Spreadsheet range.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Clears cell values from a specified range while preserving formatting and
    structure.

    !!! note "Authentication Required"

        Requires OAuth2 authentication. Use `api_resource` parameter with
        authenticated Google Sheets service.

    !!! info "Formatting Preserved"

        Only values are cleared. Formatting, borders, colors, fonts, and data
        validation rules remain intact.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        cleared_range (str): The A1 notation of the cleared range.

    ???+ example "Basic Usage"

        Clear a range of cells:

        ```python
        from langchain_google_community.sheets import SheetsClearValuesTool

        tool = SheetsClearValuesTool(api_resource=service)
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "range": "Sheet1!A1:Z100",
            }
        )
        ```

    Raises:
        ValueError: If write permissions unavailable or validation fails.
        Exception: If `spreadsheet_id` is invalid, range is malformed, or API
            errors occur.
    """

    name: str = "sheets_clear_values"

    description: str = (
        "Clear values from a range in a Google Spreadsheet. "
        "Only values are cleared; formatting and structure remain."
    )

    args_schema: Type[BaseModel] = ClearValuesSchema

    def _run(
        self,
        spreadsheet_id: str,
        range: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Clear values from a Google Spreadsheet range.

        Args:
            spreadsheet_id: ID of the spreadsheet.
            range: A1 notation range to clear.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            cleared_range (str): The A1 notation of the cleared range.

        Raises:
            ValueError: If write permissions unavailable or validation fails.
            Exception: For API errors or connection issues.
        """
        # Check write permissions (requires OAuth2)
        self._check_write_permissions()

        try:
            # Validate inputs
            spreadsheet_id = validate_spreadsheet_id(spreadsheet_id)
            range = validate_range_name(range)

            # Get the Google Sheets service
            service = self._get_service()

            # Clear the values
            result = (
                service.spreadsheets()
                .values()
                .clear(
                    spreadsheetId=spreadsheet_id,
                    range=range,
                    body={},
                )
                .execute()
            )

            return {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "cleared_range": result.get("clearedRange"),
            }

        except Exception as error:
            raise Exception(f"Error clearing sheet data: {error}") from error


# ============================================================================
# 5. BATCH UPDATE VALUES (Schema + Tool)
# ============================================================================


class BatchUpdateDataSchema(BaseModel):
    """Schema for a single range update in batch operation."""

    range: str = Field(description="The A1 notation range to update.")

    values: List[List[Any]] = Field(
        description=(
            "2D array of values for this range. "
            "Supports strings, numbers, booleans, and formulas."
        )
    )


class BatchUpdateValuesSchema(WriteBaseSchema):
    """Input schema for `SheetsBatchUpdateValuesTool`."""

    data: List[BatchUpdateDataSchema] = Field(
        description="List of range/values pairs to update."
    )


class SheetsBatchUpdateValuesTool(SheetsBaseTool):
    """Tool for batch updating multiple ranges in Google Sheets efficiently.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Updates multiple ranges in a single API call, dramatically improving
    efficiency.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        total_updated_ranges (int): Number of ranges updated.
        total_updated_cells (int): Total cells updated across all ranges.
        total_updated_rows (int): Total rows updated.
        total_updated_columns (int): Total columns updated.
        responses (list): Individual results for each range with `updated_range`
            and `updated_cells`.

    ???+ example "Basic Usage"

        Update multiple ranges in one call:

        ```python
        from langchain_google_community.sheets import SheetsBatchUpdateValuesTool

        tool = SheetsBatchUpdateValuesTool(api_resource=service)
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "data": [
                    {
                        "range": "Sheet1!G1:G3",
                        "values": [["Status"], ["Active"], ["Active"]],
                    },
                    {"range": "Sheet1!H1:H3", "values": [["Country"], ["USA"]]},
                    {"range": "Sheet1!I1:I3", "values": [["Dept"], ["Eng"]]},
                ],
            }
        )
        ```

    ??? example "Update Multiple Sheets"

        Update ranges across different sheets:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                "data": [
                    {
                        "range": "Sheet1!A1:A5",
                        "values": [["Data1"], ["Data2"], ["Data3"]],
                    },
                    {"range": "Sheet2!B1:B3", "values": [["Value1"], ["Value2"]]},
                ],
                "value_input_option": "USER_ENTERED",
            }
        )
        ```

    Raises:
        ValueError: If write permissions unavailable or validation fails.
        Exception: If `spreadsheet_id` is invalid, any range is malformed, or
            API errors occur.
    """

    name: str = "sheets_batch_update_values"

    description: str = (
        "Batch update multiple ranges in a Google Spreadsheet efficiently. "
        "Updates multiple ranges in a single API call."
    )

    args_schema: Type[BaseModel] = BatchUpdateValuesSchema

    def _run(
        self,
        spreadsheet_id: str,
        data: List[Dict[str, Any]],
        value_input_option: ValueInputOption = ValueInputOption.USER_ENTERED,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Batch update multiple ranges in a Google Spreadsheet.

        Args:
            spreadsheet_id: ID of the spreadsheet.
            data: List of range/values pairs to update.
            value_input_option: How to interpret input values.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            total_updated_ranges (int): Number of ranges updated.
            total_updated_cells (int): Total cells updated.
            total_updated_rows (int): Total rows updated.
            total_updated_columns (int): Total columns updated.
            responses (list): Individual results for each range.

        Raises:
            ValueError: If write permissions unavailable, validation fails, or
                data list is empty.
            Exception: For API errors or connection issues.
        """
        # Check write permissions (requires OAuth2)
        self._check_write_permissions()

        try:
            # Validate inputs
            spreadsheet_id = validate_spreadsheet_id(spreadsheet_id)
            # value_input_option is already validated by Pydantic (it's an Enum)

            if not data:
                raise ValueError("At least one range must be specified")

            # Build the data array with validation
            # Convert DataFilterSchema objects to dictionaries
            data_dicts = self._convert_to_dict_list(data)

            value_ranges = []
            for item_dict in data_dicts:
                range_name = item_dict.get("range")
                values = item_dict.get("values")

                # Validate range
                validate_range_name(range_name)

                value_ranges.append({"range": range_name, "values": values})

            # Get the Google Sheets service
            service = self._get_service()

            # Build the request body
            body = {"valueInputOption": value_input_option.value, "data": value_ranges}

            # Batch update the values
            result = (
                service.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )

            # Parse responses
            responses = result.get("responses", [])
            total_updated_cells = sum(
                response.get("updatedCells", 0) for response in responses
            )
            total_updated_rows = sum(
                response.get("updatedRows", 0) for response in responses
            )
            total_updated_columns = sum(
                response.get("updatedColumns", 0) for response in responses
            )

            return {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "total_updated_ranges": len(responses),
                "total_updated_cells": total_updated_cells,
                "total_updated_rows": total_updated_rows,
                "total_updated_columns": total_updated_columns,
                "responses": [
                    {
                        "updated_range": r.get("updatedRange"),
                        "updated_cells": r.get("updatedCells", 0),
                    }
                    for r in responses
                ],
            }

        except Exception as error:
            raise Exception(f"Error batch updating sheet data: {error}") from error
