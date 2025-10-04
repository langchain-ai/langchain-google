"""Read tools for Google Sheets.

This module contains all read-related tools for Google Sheets, including
base schemas, base tool class, and specific read implementations.
"""

from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, model_validator

from .base import SheetsBaseTool
from .enums import (
    DateTimeRenderOption,
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

    def _extract_simple_data_all(self, grid_data: List[dict]) -> List[List[str]]:
        """Extract a simple 2D array from ALL GridData segments, preserving order.

        This method processes ALL grid segments returned by the API, not just the
        first one. This is critical for preventing data loss when the API returns
        multiple segments (e.g., filtered reads, paginated responses).

        Args:
            grid_data: List of GridData segments from Google Sheets API

        Returns:
            List[List[str]]: Simple 2D array of values from all segments concatenated
        """
        if not grid_data:
            return []

        result: List[List[str]] = []
        for grid in grid_data:
            for row_data in grid.get("rowData", []) or []:
                row_values: List[str] = []
                for cell_data in row_data.get("values", []) or []:
                    row_values.append(self._safe_get_cell_value(cell_data))
                # Keep empty rows as [] if the API returns them
                result.append(row_values)

        return result

    def _extract_simple_data(self, grid_data: List[dict]) -> List[List[str]]:
        """Extract simple 2D array from complex GridData structure.

        Backward-compatible method name. Now processes ALL segments to prevent
        data loss.

        Args:
            grid_data: GridData from Google Sheets API

        Returns:
            List[List[str]]: Simple 2D array of values from all segments
        """
        # Delegate to the new multi-segment implementation
        return self._extract_simple_data_all(grid_data)


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
                numericise_values=True,
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "range_name": "A1:E10",
                    "convert_to_records": True,
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke(
                {"input": "Read the first 10 rows from the student data spreadsheet"}
            )

    Returns:
        Dictionary containing:
            - success (bool): Always True for successful operations
            - spreadsheet_id (str): The spreadsheet ID
            - range (str): The actual range that was read (A1 notation)
            - values (List or List[Dict]): Processed data (2D array or records)
            - major_dimension (str): The major dimension ("ROWS" or "COLUMNS")
            - render_options (Dict): Applied rendering options
            - processing_options (Dict): Applied processing options

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

    Example Response (with convert_to_records=True):
        {
            "success": True,
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "range": "Class Data!A1:E3",
            "values": [
                {
                    "Name": "Alice", "Age": 16, "Grade": "10th",
                    "Subject": "Math", "Score": 95
                },
                {
                    "Name": "Bob", "Age": 17, "Grade": "11th",
                    "Subject": "Science", "Score": 87
                }
            ],
            "major_dimension": "ROWS",
            "render_options": {
                "value_render_option": "FORMATTED_VALUE",
                "date_time_render_option": "SERIAL_NUMBER"
            },
            "processing_options": {
                "convert_to_records": True,
                "numericise_values": True
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
    ) -> Dict[str, Any]:
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
                "success": True,
                "spreadsheet_id": spreadsheet_id,
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

            return result

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
                numericise_values=True,
            )

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "ranges": ["A1:C5", "F1:H5", "Sheet2!A1:D10"],
                    "convert_to_records": True,
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Read data from multiple ranges in the spreadsheet"})

    Returns:
        Dictionary containing:
            - success (bool): Always True for successful operations
            - spreadsheet_id (str): The spreadsheet ID
            - requested_ranges (List[str]): The ranges that were requested
            - total_ranges (int): Total number of ranges processed
            - successful_ranges (int): Number of successfully processed ranges
            - failed_ranges (int): Number of failed ranges
            - results (List[Dict]): List of results for each range
            - value_render_option (str): Applied value rendering option
            - date_time_render_option (str): Applied date/time rendering option
            - major_dimension (str): Applied major dimension
            - convert_to_records (bool): Whether data was converted to records
            - numericise_values (bool): Whether values were numericised

    Performance Benefits:
        - Single API call: Reduces network overhead and rate limiting
        - Parallel processing: All ranges processed simultaneously
        - Efficient batching: Optimized for multiple data extraction scenarios
        - Consistent formatting: All ranges processed with same options

    Example Response (with convert_to_records=True):
        {
            "success": True,
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "requested_ranges": ["Class Data!A1:C3", "Class Data!F1:H3"],
            "total_ranges": 2,
            "successful_ranges": 2,
            "failed_ranges": 0,
            "results": [
                {
                    "range": "Class Data!A1:C3",
                    "data": [
                        {"Name": "Alice", "Age": 16, "Grade": "10th"},
                        {"Name": "Bob", "Age": 17, "Grade": "11th"}
                    ],
                    "error": None
                },
                {
                    "range": "Class Data!F1:H3",
                    "data": [
                        {"Subject": "Math", "Score": 95, "Teacher": "Mr. Smith"},
                        {
                            "Subject": "Science", "Score": 87,
                            "Teacher": "Ms. Johnson"
                        }
                    ],
                    "error": None
                }
            ],
            "value_render_option": "FORMATTED_VALUE",
            "date_time_render_option": "SERIAL_NUMBER",
            "major_dimension": "ROWS",
            "convert_to_records": True,
            "numericise_values": True
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
    ) -> Dict[str, Any]:
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

            # Process the response with enhanced metadata
            results = []
            successful_ranges = 0
            failed_ranges = 0
            value_ranges = response.get("valueRanges", [])

            for i, value_range in enumerate(value_ranges):
                range_name = value_range.get("range", f"range_{i}")
                values = value_range.get("values", [])
                error = None

                try:
                    processed_data = self._process_data(
                        values, convert_to_records, numericise_values
                    )
                    successful_ranges += 1
                except Exception as e:
                    processed_data = None
                    error = str(e)
                    failed_ranges += 1

                results.append(
                    {
                        "range": range_name,
                        "data": processed_data,
                        "error": error,
                    }
                )

            batch_metadata = {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "requested_ranges": ranges,
                "value_render_option": value_render_option.value,
                "date_time_render_option": date_time_render_option.value,
                "major_dimension": major_dimension.value,
                "convert_to_records": convert_to_records,
                "numericise_values": numericise_values,
                "total_ranges": len(ranges),
                "successful_ranges": successful_ranges,
                "failed_ranges": failed_ranges,
                "results": results,
            }

            return batch_metadata

        except Exception as error:
            raise Exception(f"Error batch reading sheet data: {error}") from error


# ============================================================================
# 5. FILTERED READ SHEET DATA (Schema + Tool)
# ============================================================================


class DataFilterSchema(BaseModel):
    """Schema for DataFilter used with getByDataFilter API.

    DataFilters specify which ranges or metadata to read from a spreadsheet.
    Must specify exactly ONE of: a1Range, gridRange, or developerMetadataLookup.

    Note: This is for range selection, NOT conditional filtering (like "score > 50").
    For conditional filtering, use Filter Views via the batchUpdate API.

    See: https://developers.google.com/sheets/api/reference/rest/v4/DataFilter
    """

    a1Range: Optional[str] = Field(
        None,
        description=(
            "A1 notation range to read (e.g., 'Sheet1!A1:D5', 'NamedRange'). "
            "This is the most common way to specify a range."
        ),
    )
    gridRange: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Grid coordinates to read. Example: "
            "{'sheetId': 0, 'startRowIndex': 0, 'endRowIndex': 10, "
            "'startColumnIndex': 0, 'endColumnIndex': 5}"
        ),
    )
    developerMetadataLookup: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Developer metadata lookup to select ranges. "
            "Advanced feature for ranges tagged with metadata."
        ),
    )

    @model_validator(mode="after")
    def check_one_filter_type(self) -> "DataFilterSchema":
        """Validate that exactly one filter type is specified."""
        count = sum(
            [
                bool(self.a1Range),
                bool(self.gridRange),
                bool(self.developerMetadataLookup),
            ]
        )
        if count != 1:
            raise ValueError(
                "Must specify exactly one of: a1Range, gridRange, or "
                "developerMetadataLookup"
            )
        return self


class FilteredReadSheetDataSchema(ReadBaseSchema):
    """Input schema for reading data using DataFilters (getByDataFilter API)."""

    data_filters: List[DataFilterSchema] = Field(
        ...,
        description=(
            "List of DataFilters specifying which ranges to read. "
            "Each filter selects a range using a1Range, gridRange, or metadata lookup."
        ),
    )
    include_grid_data: bool = Field(
        default=False,
        description=(
            "Whether to include detailed grid data (cell properties, formatting). "
            "Note: This can significantly increase response size."
        ),
    )


class SheetsFilteredReadDataTool(BaseReadTool):
    """Tool that reads data from Google Sheets using DataFilters (getByDataFilter API).

    This tool reads data from spreadsheets using the getByDataFilter API, which
    allows you to specify ranges using A1 notation, grid coordinates, or developer
    metadata. It also provides detailed cell formatting and properties when
    include_grid_data=True.

    Note: This tool is for RANGE SELECTION, not conditional filtering like
    "score > 50". For conditional filtering, create a Filter View using the
    Sheets UI or batchUpdate API.

    Use cases:
    - Read multiple ranges in a single API call
    - Get detailed cell formatting and properties (gridData)
    - Select ranges by developer metadata tags
    - Use grid coordinates instead of A1 notation

    Requires OAuth2 authentication for full functionality.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsFilteredReadDataTool

            tool = SheetsFilteredReadDataTool(
                credentials_path="path/to/credentials.json",
                include_grid_data=True,
                convert_to_records=True,
                numericise_values=True,
            )

    Invoke directly:
        .. code-block:: python

            # Example 1: Read using A1 notation
            result = tool.run(
                {
                    "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "data_filters": [{"a1Range": "Class Data!A1:E10"}],
                    "include_grid_data": True,
                }
            )

            # Example 2: Read using grid coordinates
            result = tool.run(
                {
                    "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "data_filters": [
                        {
                            "gridRange": {
                                "sheetId": 0,
                                "startRowIndex": 0,
                                "endRowIndex": 10,
                                "startColumnIndex": 0,
                                "endColumnIndex": 5,
                            }
                        }
                    ],
                    "include_grid_data": True,
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke(
                {"input": "Read the range Class Data!A1:E10 with formatting details"}
            )

    Returns:
        Dictionary containing:
            - success (bool): Always True for successful operations
            - spreadsheet_id (str): The spreadsheet ID
            - properties (Dict): Spreadsheet-level properties
            - sheets (List[Dict]): List of sheets with filtered data
                Each sheet contains:
                - properties (Dict): Sheet properties (title, sheetId, etc.)
                - data (List): List of data segments
                    Each segment is either List[List] or List[Dict]
                    depending on convert_to_records setting

    DataFilter Types:
        - a1Range: Most common, uses A1 notation (e.g., "Sheet1!A1:D5")
        - gridRange: Uses grid coordinates (sheetId, row/column indices)
        - developerMetadataLookup: Selects ranges tagged with metadata

    Authentication Requirements:
        - Requires OAuth2 credentials (not API key)
        - Full access to spreadsheet data
        - Supports private and shared spreadsheets

    Example Response (with convert_to_records=True):
        {
            "success": True,
            "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            "properties": {
                "title": "Student Data",
                "locale": "en_US",
                "timeZone": "America/New_York"
            },
            "sheets": [
                {
                    "properties": {
                        "sheetId": 0,
                        "title": "Students",
                        "index": 0,
                        "sheetType": "GRID"
                    },
                    "data": [
                        [
                            {
                                "Name": "Alice", "Score": 95,
                                "Subject": "Math", "Grade": "A"
                            },
                            {
                                "Name": "Charlie", "Score": 87,
                                "Subject": "Science", "Grade": "B"
                            }
                        ]
                    ]
                }
            ]
        }

    Raises:
        ValueError: If spreadsheet_id is invalid or data_filters is empty
        Exception: For API errors, authentication issues, or connection problems
    """

    name: str = "sheets_get_by_data_filter"
    description: str = (
        "Read data from Google Sheets using DataFilters (getByDataFilter API). "
        "Select ranges using A1 notation, grid coordinates, or developer metadata. "
        "Optionally include detailed cell formatting and properties (gridData). "
        "Useful for reading multiple ranges or ranges with specific metadata tags. "
        "Requires OAuth2 authentication."
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
    ) -> Dict[str, Any]:
        """Read data from Google Sheets with filtering."""
        try:
            # Validate inputs
            validate_spreadsheet_id(spreadsheet_id)
            if not data_filters:
                raise ValueError("At least one data filter must be specified")

            # Get the service
            service = self._get_service()

            # Convert DataFilterSchema objects to dictionaries
            data_filters_dict = self._convert_to_dict_list(data_filters)

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

            # Add success field
            result["success"] = True

            return result

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
