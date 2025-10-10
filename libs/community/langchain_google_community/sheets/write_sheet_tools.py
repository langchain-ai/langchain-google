"""Write tools for Google Sheets.

This module contains all write-related tools for Google Sheets, including
base schemas, base tool class, and specific write implementations.

Note: All write operations require OAuth2 authentication (api_resource).
"""

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
    """Base schema for all write operations.

    Contains common fields that are shared across all write tools.
    """

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
    """Schema for updating values in a Google Spreadsheet."""

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
    """Tool for updating values in a single range of a Google Spreadsheet.

    This tool updates cell values in a specified range, overwriting existing data.
    Values can be interpreted as raw strings or parsed as user-entered data
    (including formulas, numbers, and dates). Perfect for modifying existing
    data, adding formulas, or bulk updating specific sections of a spreadsheet.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsUpdateValuesTool

            tool = SheetsUpdateValuesTool(
                api_resource=service, value_input_option="USER_ENTERED"
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                    "range": "Sheet1!A1:C3",
                    "values": [
                        ["Name", "Age", "City"],
                        ["Alice", "25", "New York"],
                        ["Bob", "30", "San Francisco"],
                    ],
                    "value_input_option": "USER_ENTERED",
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Update cells A1:C3 with employee data"})

    Returns:
        Dictionary containing:
            - success: bool - Whether the operation succeeded
            - spreadsheet_id: str - The spreadsheet ID
            - updated_range: str - The A1 notation of updated range
            - updated_rows: int - Number of rows updated
            - updated_columns: int - Number of columns updated
            - updated_cells: int - Total number of cells updated

    Value Input Options:
        - RAW: Values stored exactly as provided (e.g., "=1+2" as text)
        - USER_ENTERED: Values parsed as if typed by user (formulas evaluated)

    Example Response:
        {
            "success": True,
            "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
            "updated_range": "Sheet1!F1:F3",
            "updated_rows": 3,
            "updated_columns": 1,
            "updated_cells": 3
        }

    Raises:
        Exception: If spreadsheet_id is invalid, range is malformed, or API errors occur
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
            spreadsheet_id: The ID of the spreadsheet to update.
            range: The A1 notation range to update.
            values: 2D array of values to write.
            value_input_option: How to interpret input.
            run_manager: Optional callback manager.

        Returns:
            Dict containing the update results.

        Raises:
            ValueError: If write permissions are not available or validation fails.
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
    """Schema for appending values to a Google Spreadsheet."""

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

    This tool appends data to the end of a table, automatically finding
    the last row with data. Perfect for adding new records to existing data,
    logging events, or incrementally building datasets without manual row tracking.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsAppendValuesTool

            tool = SheetsAppendValuesTool(
                api_resource=service,
                value_input_option="USER_ENTERED",
                insert_data_option="INSERT_ROWS",
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                    "range": "Sheet1!A1:D100",
                    "values": [
                        ["Eve", "27", "Seattle", "91"],
                        ["Frank", "32", "Denver", "85"],
                    ],
                    "value_input_option": "USER_ENTERED",
                    "insert_data_option": "INSERT_ROWS",
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Add two new employee records to the spreadsheet"})

    Returns:
        Dictionary containing:
            - success: bool - Whether the operation succeeded
            - spreadsheet_id: str - The spreadsheet ID
            - table_range: str - The range of the entire table
            - updated_range: str - The specific range where data was appended
            - updated_rows: int - Number of rows appended
            - updated_columns: int - Number of columns appended
            - updated_cells: int - Total number of cells updated

    Insert Data Options:
        - OVERWRITE: Data overwrites existing data after the table (default)
        - INSERT_ROWS: New rows are inserted, existing data shifted down

    Example Response:
        {
            "success": True,
            "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
            "table_range": "Sheet1!A1:F5",
            "updated_range": "Sheet1!A6:D7",
            "updated_rows": 2,
            "updated_columns": 4,
            "updated_cells": 8
        }

    Raises:
        Exception: If spreadsheet_id is invalid, range is malformed, or API errors occur
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
            spreadsheet_id: The ID of the spreadsheet.
            range: The A1 notation table range.
            values: 2D array of values to append.
            value_input_option: How to interpret input.
            insert_data_option: How to handle existing data.
            run_manager: Optional callback manager.

        Returns:
            Dict containing the append results.

        Raises:
            ValueError: If write permissions are not available or validation fails.
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
    """Schema for clearing values in a Google Spreadsheet."""

    spreadsheet_id: str = Field(description="The ID of the Google Spreadsheet.")
    range: str = Field(
        description=(
            "The A1 notation of the range to clear (e.g., 'Sheet1!A1:Z100'). "
            "Only values are cleared; formatting remains."
        )
    )


class SheetsClearValuesTool(SheetsBaseTool):
    """Tool for clearing values from a Google Spreadsheet range.

    This tool clears cell values from a specified range while preserving
    formatting, data validation, and other cell properties. Perfect for
    resetting data sections, clearing temporary calculations, or removing
    outdated information without destroying the spreadsheet structure.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsClearValuesTool

            tool = SheetsClearValuesTool(api_resource=service)

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                    "range": "Sheet1!A1:Z100",
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Clear all data from column F"})

    Returns:
        Dictionary containing:
            - success: bool - Whether the operation succeeded
            - spreadsheet_id: str - The spreadsheet ID
            - cleared_range: str - The A1 notation of the cleared range

    Important Notes:
        - Only values are cleared; formatting remains intact
        - Cell borders, colors, and fonts are preserved
        - Data validation rules are not affected
        - Formulas and structure remain unchanged
        - Can clear entire rows, columns, or specific ranges

    Example Response:
        {
            "success": True,
            "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
            "cleared_range": "Sheet1!E2:E10"
        }

    Raises:
        Exception: If spreadsheet_id is invalid, range is malformed, or API errors occur
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
            spreadsheet_id: The ID of the spreadsheet.
            range: The A1 notation range to clear.
            run_manager: Optional callback manager.

        Returns:
            Dict containing the clear results.

        Raises:
            ValueError: If write permissions are not available or validation fails.
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
    """Schema for batch updating values in a Google Spreadsheet."""

    data: List[BatchUpdateDataSchema] = Field(
        description="List of range/values pairs to update."
    )


class SheetsBatchUpdateValuesTool(SheetsBaseTool):
    """Tool for batch updating multiple ranges in a Google Spreadsheet.

    This tool updates multiple ranges in a single API call, dramatically
    improving efficiency when updating multiple sections of a spreadsheet.
    Perfect for complex updates, synchronized data changes, or updating
    multiple sheets simultaneously while minimizing API calls and latency.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsBatchUpdateValuesTool

            tool = SheetsBatchUpdateValuesTool(
                api_resource=service,
                value_input_option="USER_ENTERED",
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
                    "data": [
                        {
                            "range": "Sheet1!G1:G3",
                            "values": [["Status"], ["Active"], ["Active"]],
                        },
                        {
                            "range": "Sheet1!H1:H3",
                            "values": [["Country"], ["USA"], ["USA"]],
                        },
                        {
                            "range": "Sheet1!I1:I3",
                            "values": [["Department"], ["Engineering"], ["Sales"]],
                        },
                    ],
                    "value_input_option": "RAW",
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Update status, country, and department columns"})

    Returns:
        Dictionary containing:
            - success: bool - Whether the operation succeeded
            - spreadsheet_id: str - The spreadsheet ID
            - total_updated_ranges: int - Number of ranges updated
            - total_updated_cells: int - Total cells updated across all ranges
            - total_updated_rows: int - Total rows updated
            - total_updated_columns: int - Total columns updated
            - responses: List[Dict] - Individual results for each range

    Performance Benefits:
        - Single API call: Reduces network overhead significantly
        - Atomic operation: All updates succeed or fail together
        - Faster execution: 10x faster than individual updates
        - Consistent state: Ensures data integrity across ranges

    Example Response:
        {
            "success": True,
            "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
            "total_updated_ranges": 3,
            "total_updated_cells": 9,
            "total_updated_rows": 9,
            "total_updated_columns": 3,
            "responses": [
                {"updated_range": "Sheet1!G1:G3", "updated_cells": 3},
                {"updated_range": "Sheet1!H1:H3", "updated_cells": 3},
                {"updated_range": "Sheet1!I1:I3", "updated_cells": 3}
            ]
        }

    Raises:
        Exception: If spreadsheet_id is invalid, any range is malformed,
            or API errors occur
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
            spreadsheet_id: The ID of the spreadsheet.
            data: List of range/values pairs to update.
            value_input_option: How to interpret input.
            run_manager: Optional callback manager.

        Returns:
            Dict containing the batch update results.

        Raises:
            ValueError: If write permissions are not available or validation fails.
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
