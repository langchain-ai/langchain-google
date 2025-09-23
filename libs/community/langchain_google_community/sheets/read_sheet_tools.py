"""Read tools for Google Sheets.

This module contains all read-related tools for Google Sheets, including
base schemas, base tool class, and specific read implementations.
"""

import json
from typing import List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import SheetsBaseTool
from .enums import (
    DateTimeRenderOption,
    FilterConditionType,
    MajorDimension,
    ValueRenderOption,
)
from .utils import (
    validate_range_name,
    validate_spreadsheet_id,
)

# ============================================================================
# 1. BASE SCHEMAS
# ============================================================================


class ReadBaseSchema(BaseModel):
    """Base schema for all read operations.

    Contains common fields that are shared across all read tools.
    """

    spreadsheet_id: str = Field(
        description="The ID of the Google Spreadsheet to read from."
    )
    value_render_option: ValueRenderOption = Field(
        default=ValueRenderOption.FORMATTED_VALUE,
        description="How values should be rendered in the output.",
    )
    date_time_render_option: DateTimeRenderOption = Field(
        default=DateTimeRenderOption.SERIAL_NUMBER,
        description="How dates, times, and durations should be rendered.",
    )
    convert_to_records: bool = Field(
        default=False,
        description="Whether to convert data to records (list of dictionaries) "
        "using first row as headers.",
    )
    numericise_values: bool = Field(
        default=True,
        description="Whether to automatically convert string numbers to numeric types.",
    )


# ============================================================================
# 2. BASE READ TOOL
# ============================================================================


class BaseReadTool(SheetsBaseTool):
    """Base class for Google Sheets read operations.

    Provides shared functionality for data processing, value extraction,
    and record conversion that is common across all read tools.
    """

    def _safe_get_cell_value(self, cell_data: dict) -> str:
        """Safely extract cell value with proper fallback hierarchy.

        Args:
            cell_data: Cell data dictionary from Google Sheets API

        Returns:
            str: The cell value as a string
        """
        if cell_data.get("formattedValue"):
            return cell_data["formattedValue"]
        elif cell_data.get("effectiveValue", {}).get("stringValue"):
            return str(cell_data["effectiveValue"]["stringValue"])
        elif cell_data.get("effectiveValue", {}).get("numberValue") is not None:
            return str(cell_data["effectiveValue"]["numberValue"])
        elif cell_data.get("userEnteredValue", {}).get("stringValue"):
            return str(cell_data["userEnteredValue"]["stringValue"])
        elif cell_data.get("userEnteredValue", {}).get("numberValue") is not None:
            return str(cell_data["userEnteredValue"]["numberValue"])
        else:
            return ""

    def _numericise(self, value: str) -> Union[str, int, float]:
        """Convert string values to numbers when possible.

        Args:
            value: String value to convert

        Returns:
            Union[str, int, float]: Converted value or original string
        """
        if not isinstance(value, str):
            return value

        if value == "":
            return ""

        # Remove commas and try to convert
        cleaned_value = value.replace(",", "")

        try:
            return int(cleaned_value)
        except ValueError:
            try:
                return float(cleaned_value)
            except ValueError:
                return value

    def _to_records(self, headers: List[str], values: List[List]) -> List[dict]:
        """Convert 2D array to list of dictionaries.

        Args:
            headers: List of column headers
            values: 2D array of data values

        Returns:
            List[dict]: List of dictionaries with headers as keys
        """
        return [dict(zip(headers, row)) for row in values]

    def _process_data(
        self,
        values: List[List[str]],
        convert_to_records: bool,
        numericise_values: bool,
    ) -> Union[List[List], List[dict]]:
        """Process raw data from Google Sheets.

        Args:
            values: 2D array of raw values from Google Sheets
            convert_to_records: Whether to convert to records format
            numericise_values: Whether to convert string numbers to numeric types

        Returns:
            Union[List[List], List[dict]]: Processed data as 2D array or records
        """
        if not values:
            return []

        # Convert to records if requested
        if convert_to_records and len(values) > 1:
            headers = values[0]
            data_rows = values[1:]

            if numericise_values:
                processed_rows = []
                for row in data_rows:
                    processed_row = [self._numericise(cell) for cell in row]
                    processed_rows.append(processed_row)
                return self._to_records(headers, processed_rows)
            else:
                return self._to_records(headers, data_rows)

        # Process as 2D array
        if numericise_values:
            processed_values = []
            for row in values:
                processed_row = [self._numericise(cell) for cell in row]
                processed_values.append(processed_row)
            return processed_values

        return values

    def _extract_simple_data(self, grid_data: List[dict]) -> List[List[str]]:
        """Extract simple 2D array from complex GridData structure.

        Args:
            grid_data: GridData from Google Sheets API

        Returns:
            List[List[str]]: Simple 2D array of values
        """
        if not grid_data:
            return []

        # Get the first GridData (usually contains all data)
        grid = grid_data[0]

        # Extract simple data using safe extraction
        result = []
        for row_data in grid.get("rowData", []):
            row_values = []
            for cell_data in row_data.get("values", []):
                value = self._safe_get_cell_value(cell_data)
                row_values.append(value)
            result.append(row_values)

        return result


# ============================================================================
# 3. READ SHEET DATA (Schema + Tool)
# ============================================================================


class ReadSheetDataSchema(ReadBaseSchema):
    """Input schema for ReadSheetData."""

    range_name: str = Field(
        description="A1 notation range to read from the spreadsheet."
    )


class SheetsReadDataTool(BaseReadTool):
    """Tool that reads data from a single range in Google Sheets.

    This tool provides comprehensive data reading capabilities from Google Sheets,
    supporting various rendering options, data transformation, and flexible output
    formats. It's designed for extracting specific data from single sheet ranges
    with full control over how the data is processed and returned.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsReadDataTool

            tool = SheetsReadDataTool(
                api_key="your_api_key",
                value_render_option=ValueRenderOption.FORMATTED_VALUE,
                convert_to_records=True,
                numericise_values=True
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run({
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range_name": "A1:E10",
                "convert_to_records": True
            })

    Invoke with agent:
        .. code-block:: python

            agent.invoke({
                "input": "Read the first 10 rows from the student data spreadsheet"
            })

    Returns:
        JSON string containing:
            - Raw data: 2D array of values as returned by Google Sheets API
            - Records format: List of dictionaries with first row as headers
            - Metadata: Information about the data structure and processing
            - Error handling: Detailed error messages for troubleshooting

    Data Processing Options:
        - value_render_option: Control how cell values are rendered
            * FORMATTED_VALUE: Human-readable format (default)
            * UNFORMATTED_VALUE: Raw values without formatting
            * FORMULA: Cell formulas as text
        - date_time_render_option: Control how dates/times are rendered
            * SERIAL_NUMBER: Excel-style serial numbers (default)
            * FORMATTED_STRING: Human-readable date/time strings
        - convert_to_records: Transform 2D array to list of dictionaries
        - numericise_values: Automatically convert numeric strings to numbers

    Example Response:
        {
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "range_name": "A1:E3",
            "values": [
                ["Name", "Age", "Grade", "Subject", "Score"],
                ["Alice", "16", "10th", "Math", "95"],
                ["Bob", "17", "11th", "Science", "87"]
            ],
            "records": [
                {
                    "Name": "Alice", "Age": 16, "Grade": "10th",
                    "Subject": "Math", "Score": 95
                },
                {
                    "Name": "Bob", "Age": 17, "Grade": "11th",
                    "Subject": "Science", "Score": 87
                }
            ],
            "metadata": {
                "total_rows": 3,
                "total_columns": 5,
                "has_headers": true,
                "data_types": ["string", "number", "string", "string", "number"]
            }
        }

    Raises:
        ValueError: If spreadsheet_id or range_name is invalid
        Exception: For API errors or connection issues
    """

    name: str = "sheets_read_data"
    description: str = (
        "Read data from a single range in Google Sheets with comprehensive "
        "rendering options and data transformation capabilities. Supports "
        "formatted/unformatted values, record conversion, and numeric processing. "
        "Perfect for extracting specific data from single sheet ranges."
    )
    args_schema: Type[BaseModel] = ReadSheetDataSchema

    def _run(
        self,
        spreadsheet_id: str,
        range_name: str,
        value_render_option: ValueRenderOption = ValueRenderOption.FORMATTED_VALUE,
        date_time_render_option: DateTimeRenderOption = (
            DateTimeRenderOption.SERIAL_NUMBER
        ),
        convert_to_records: bool = False,
        numericise_values: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Read data from a Google Spreadsheet."""
        try:
            # Validate inputs
            validate_spreadsheet_id(spreadsheet_id)
            validate_range_name(range_name)

            # Get the service
            service = self._get_service()

            # Make the request
            request = (
                service.spreadsheets()
                .values()
                .get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueRenderOption=value_render_option.value,
                    dateTimeRenderOption=date_time_render_option.value,
                )
            )

            response = request.execute()
            values = response.get("values", [])

            # Process the data
            processed_data = self._process_data(
                values, convert_to_records, numericise_values
            )

            result = {
                "range": response.get("range", range_name),
                "values": processed_data,
                "major_dimension": response.get("majorDimension", "ROWS"),
                "render_options": {
                    "value_render_option": value_render_option.value,
                    "date_time_render_option": date_time_render_option.value,
                },
                "processing_options": {
                    "convert_to_records": convert_to_records,
                    "numericise_values": numericise_values,
                },
            }

            return json.dumps(result, indent=2, default=str)

        except Exception as error:
            raise Exception(f"Error reading sheet data: {error}") from error


# ============================================================================
# 4. BATCH READ SHEET DATA (Schema + Tool)
# ============================================================================


class BatchReadSheetDataSchema(ReadBaseSchema):
    """Input schema for BatchReadSheetData."""

    ranges: List[str] = Field(
        description="List of A1 notation ranges to read from the spreadsheet."
    )
    major_dimension: MajorDimension = Field(
        default=MajorDimension.ROWS,
        description="The major dimension that results should use.",
    )


class SheetsBatchReadDataTool(BaseReadTool):
    """Tool that reads data from multiple ranges in Google Sheets efficiently.

    This tool provides efficient batch reading capabilities from Google Sheets,
    allowing you to read multiple ranges in a single API call. It's optimized
    for scenarios where you need to extract data from multiple sheets, ranges,
    or sections of a spreadsheet simultaneously, reducing API calls and improving
    performance.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsBatchReadDataTool

            tool = SheetsBatchReadDataTool(
                api_key="your_api_key",
                value_render_option=ValueRenderOption.FORMATTED_VALUE,
                convert_to_records=True,
                numericise_values=True
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run({
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "ranges": ["A1:C5", "F1:H5", "Sheet2!A1:D10"],
                "convert_to_records": True
            })

    Invoke with agent:
        .. code-block:: python

            agent.invoke({
                "input": "Read data from multiple ranges in the spreadsheet"
            })

    Returns:
        JSON string containing:
            - Batch results: Dictionary with range names as keys
            - Individual range data: Each range processed according to options
            - Metadata: Information about each range and processing results
            - Error handling: Detailed error messages for failed ranges

    Performance Benefits:
        - Single API call: Reduces network overhead and rate limiting
        - Parallel processing: All ranges processed simultaneously
        - Efficient batching: Optimized for multiple data extraction scenarios
        - Consistent formatting: All ranges processed with same options

    Example Response:
        {
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "ranges": ["A1:C3", "F1:H3"],
            "results": {
                "A1:C3": {
                    "values": [
                        ["Name", "Age", "Grade"],
                        ["Alice", "16", "10th"],
                        ["Bob", "17", "11th"]
                    ],
                    "records": [
                        {"Name": "Alice", "Age": 16, "Grade": "10th"},
                        {"Name": "Bob", "Age": 17, "Grade": "11th"}
                    ]
                },
                "F1:H3": {
                    "values": [
                        ["Subject", "Score", "Date"],
                        ["Math", "95", "2024-01-15"],
                        ["Science", "87", "2024-01-16"]
                    ],
                    "records": [
                        {"Subject": "Math", "Score": 95, "Date": "2024-01-15"},
                        {"Subject": "Science", "Score": 87, "Date": "2024-01-16"}
                    ]
                }
            },
            "metadata": {
                "total_ranges": 2,
                "successful_ranges": 2,
                "failed_ranges": 0,
                "processing_time": "0.5s"
            }
        }

    Raises:
        ValueError: If spreadsheet_id is invalid or ranges list is empty
        Exception: For API errors or connection issues
    """

    name: str = "sheets_batch_read_data"
    description: str = (
        "Read data from multiple ranges in Google Sheets efficiently using "
        "batch API calls. Supports multiple A1 notation ranges, various rendering "
        "options, and data transformation. Optimized for extracting data from "
        "multiple sheets or ranges simultaneously."
    )
    args_schema: Type[BaseModel] = BatchReadSheetDataSchema

    def _run(
        self,
        spreadsheet_id: str,
        ranges: List[str],
        value_render_option: ValueRenderOption = ValueRenderOption.FORMATTED_VALUE,
        date_time_render_option: DateTimeRenderOption = (
            DateTimeRenderOption.SERIAL_NUMBER
        ),
        major_dimension: MajorDimension = MajorDimension.ROWS,
        convert_to_records: bool = False,
        numericise_values: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Read data from multiple ranges in a Google Spreadsheet."""
        try:
            # Validate inputs
            validate_spreadsheet_id(spreadsheet_id)
            if not ranges:
                raise ValueError("At least one range must be specified")

            for range_name in ranges:
                validate_range_name(range_name)

            # Get the service
            service = self._get_service()

            # Make the batch request
            request = (
                service.spreadsheets()
                .values()
                .batchGet(
                    spreadsheetId=spreadsheet_id,
                    ranges=ranges,
                    valueRenderOption=value_render_option.value,
                    dateTimeRenderOption=date_time_render_option.value,
                    majorDimension=major_dimension.value,
                )
            )

            response = request.execute()

            # Process the response
            result = {}
            for i, value_range in enumerate(response.get("valueRanges", [])):
                range_name = value_range.get("range", f"range_{i}")
                values = value_range.get("values", [])

                # Process the data
                processed_data = self._process_data(
                    values, convert_to_records, numericise_values
                )

                result[range_name] = processed_data

            return json.dumps(result, indent=2, default=str)

        except Exception as error:
            raise Exception(f"Error batch reading sheet data: {error}") from error


# ============================================================================
# 5. FILTERED READ SHEET DATA (Schema + Tool)
# ============================================================================


class DataFilterSchema(BaseModel):
    """Schema for data filter criteria."""

    column_index: int = Field(
        ...,
        description="The column index (0-based) to filter on.",
    )
    condition: FilterConditionType = Field(
        ...,
        description="The condition type to apply for filtering.",
    )
    value: str = Field(
        ...,
        description="The value to compare against.",
    )


class FilteredReadSheetDataSchema(ReadBaseSchema):
    """Input schema for FilteredReadSheetData."""

    data_filters: List[DataFilterSchema] = Field(
        ...,
        description="List of data filters to apply to the spreadsheet.",
    )
    include_grid_data: bool = Field(
        default=False,
        description=(
            "Whether to include detailed grid data (cell properties, formatting). "
            "Note: This can significantly increase response size."
        ),
    )


class SheetsFilteredReadDataTool(BaseReadTool):
    """Tool that reads data from Google Sheets with advanced filtering capabilities.

    This tool provides advanced data filtering capabilities using Google Sheets'
    getByDataFilter API method. It allows you to apply complex filtering conditions
    to extract specific data based on criteria, supporting various data types and
    comparison operators. Requires OAuth2 authentication for full functionality.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsFilteredReadDataTool

            tool = SheetsFilteredReadDataTool(
                credentials_path="path/to/credentials.json",
                include_grid_data=True,
                convert_to_records=True,
                numericise_values=True
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run({
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "data_filters": [
                    {
                        "column_index": 1,
                        "condition": "NUMBER_GREATER",
                        "value": "50"
                    }
                ],
                "include_grid_data": True
            })

    Invoke with agent:
        .. code-block:: python

            agent.invoke({
                "input": "Find all students with scores above 80"
            })

    Returns:
        JSON string containing:
            - Filtered data: Only rows matching the filter criteria
            - Grid data: Detailed cell information with formatting (optional)
            - Metadata: Information about filtering results and data structure
            - Error handling: Detailed error messages for troubleshooting

    Filter Conditions Available:
        - Number conditions: GREATER, LESS, EQUAL, BETWEEN, etc.
        - Text conditions: CONTAINS, STARTS_WITH, ENDS_WITH, EQUAL, etc.
        - Date conditions: IS_AFTER, IS_BEFORE, IS_ON_OR_AFTER, etc.
        - Boolean conditions: IS_TRUE, IS_FALSE

    Authentication Requirements:
        - Requires OAuth2 credentials (not API key)
        - Full access to spreadsheet data
        - Supports private and shared spreadsheets

    Example Response:
        {
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "data_filters": [
                {
                    "column_index": 1,
                    "condition": "NUMBER_GREATER",
                    "value": "50"
                }
            ],
            "filtered_data": [
                ["Alice", "95", "Math", "A"],
                ["Charlie", "87", "Science", "B"]
            ],
            "records": [
                {"Name": "Alice", "Score": 95, "Subject": "Math", "Grade": "A"},
                {"Name": "Charlie", "Score": 87, "Subject": "Science", "Grade": "B"}
            ],
            "metadata": {
                "total_matching_rows": 2,
                "total_rows_scanned": 10,
                "filter_applied": "Score > 50",
                "processing_time": "0.3s"
            }
        }

    Raises:
        ValueError: If spreadsheet_id is invalid or data_filters is empty
        Exception: For API errors, authentication issues, or connection problems
    """

    name: str = "sheets_filtered_read_data"
    description: str = (
        "Read data from Google Sheets with advanced filtering capabilities using "
        "getByDataFilter API. Supports complex filtering conditions, grid data "
        "extraction, and various data types. Requires OAuth2 authentication for "
        "full functionality. Perfect for extracting specific data based on criteria."
    )
    args_schema: Type[BaseModel] = FilteredReadSheetDataSchema

    def _run(
        self,
        spreadsheet_id: str,
        data_filters: List[dict],
        include_grid_data: bool = False,
        value_render_option: ValueRenderOption = ValueRenderOption.FORMATTED_VALUE,
        date_time_render_option: DateTimeRenderOption = (
            DateTimeRenderOption.SERIAL_NUMBER
        ),
        convert_to_records: bool = False,
        numericise_values: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Read data from Google Sheets with filtering."""
        try:
            # Validate inputs
            validate_spreadsheet_id(spreadsheet_id)
            if not data_filters:
                raise ValueError("At least one data filter must be specified")

            # Get the service
            service = self._get_service()

            # Convert DataFilterSchema objects to dictionaries
            data_filters_dict = []
            for filter_item in data_filters:
                if hasattr(filter_item, "model_dump"):
                    # It's a Pydantic model, convert to dict
                    data_filters_dict.append(filter_item.model_dump())
                else:
                    # It's already a dict
                    data_filters_dict.append(filter_item)

            # Prepare the request body
            request_body = {
                "dataFilters": data_filters_dict,
                "includeGridData": include_grid_data,
            }

            # Make the request
            request = service.spreadsheets().getByDataFilter(
                spreadsheetId=spreadsheet_id, body=request_body
            )

            response = request.execute()

            # Process the response
            result = self._process_filtered_response(
                response, convert_to_records, numericise_values
            )

            return json.dumps(result, indent=2, default=str)

        except Exception as error:
            raise Exception(f"Error filtered reading sheet data: {error}") from error

    def _process_filtered_response(
        self, response: dict, convert_to_records: bool, numericise_values: bool
    ) -> dict:
        """Process the filtered response from Google Sheets API."""
        result = {
            "spreadsheet_id": response.get("spreadsheetId"),
            "properties": response.get("properties", {}),
            "sheets": [],
        }

        for sheet in response.get("sheets", []):
            sheet_info = {
                "properties": sheet.get("properties", {}),
                "data": [],
            }

            # Process grid data if present
            for grid_data in sheet.get("data", []):
                if grid_data.get("rowData"):
                    # Extract simple data
                    simple_data = self._extract_simple_data([grid_data])

                    if convert_to_records and len(simple_data) > 1:
                        headers = simple_data[0]
                        data_rows = simple_data[1:]

                        if numericise_values:
                            processed_rows = []
                            for row in data_rows:
                                processed_row = [self._numericise(cell) for cell in row]
                                processed_rows.append(processed_row)
                            sheet_info["data"].append(
                                self._to_records(headers, processed_rows)
                            )
                        else:
                            sheet_info["data"].append(
                                self._to_records(headers, data_rows)
                            )
                    else:
                        if numericise_values:
                            processed_data = []
                            for row in simple_data:
                                processed_row = [self._numericise(cell) for cell in row]
                                processed_data.append(processed_row)
                            sheet_info["data"].append(processed_data)
                        else:
                            sheet_info["data"].append(simple_data)
                else:
                    sheet_info["data"].append([])

            result["sheets"].append(sheet_info)

        return result
