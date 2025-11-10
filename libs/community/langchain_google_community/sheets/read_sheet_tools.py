"""Tools for reading data from Google Sheets."""

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
    """Base schema for read operations with common fields."""

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

    Provides shared functionality for data processing, value extraction, and record
    conversion that is common across all read tools.
    """

    def _numericise(self, value: str) -> Union[str, int, float]:
        """Convert string values to numbers when possible.

        Args:
            value: String value to convert

        Returns:
            Converted value or original string.
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
            List of dictionaries with headers as keys.
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
            Processed data as 2D array or records.
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
            Simple 2D array of values from all segments concatenated.
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
            2D array of values from all segments.
        """
        # Delegate to the new multi-segment implementation
        return self._extract_simple_data_all(grid_data)


# ============================================================================
# 3. READ SHEET DATA (Schema + Tool)
# ============================================================================


class ReadSheetDataSchema(ReadBaseSchema):
    """Input schema for `SheetsReadDataTool`."""

    range_name: str = Field(
        description="A1 notation range to read from the spreadsheet."
    )


class SheetsReadDataTool(BaseReadTool):
    """Tool for reading data from a single range in Google Sheets.

    Inherits from
    [`BaseReadTool`][langchain_google_community.sheets.read_sheet_tools.BaseReadTool].

    Reads data from a single range with support for various rendering options and data
    transformations.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        range (str): The actual range that was read (A1 notation).
        values (list): Processed data (2D array or list of dictionaries).
        major_dimension (str): The major dimension ('ROWS' or 'COLUMNS').
        render_options (dict): Applied rendering options.
        processing_options (dict): Applied processing options.

    ???+ example "Basic Usage"

        Read data from a range:

        ```python
        from langchain_google_community.sheets import SheetsReadDataTool

        tool = SheetsReadDataTool(api_key="your_api_key")
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range_name": "A1:E10",
            }
        )
        print(result["values"])
        ```

    ??? example "Convert to Records"

        Convert 2D array to list of dictionaries using first row as headers:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range_name": "A1:E10",
                "convert_to_records": True,
            }
        )
        # Returns: [{"Name": "Alice", "Age": 25, ...}, ...]
        ```

    ??? example "Custom Rendering Options"

        Get unformatted values and formulas:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "range_name": "A1:E10",
                "value_render_option": "UNFORMATTED_VALUE",
            }
        )
        ```

    Raises:
        ValueError: If `spreadsheet_id` or `range_name` is invalid.
        Exception: For API errors or connection issues.
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
        """Read data from a single range in Google Sheets.

        Args:
            spreadsheet_id: ID of the spreadsheet to read from.
            range_name: A1 notation range (e.g., `'A1:E10'` or `'Sheet1!A1:D5'`).
            value_render_option: How values should be rendered.
            date_time_render_option: How dates/times should be rendered.
            convert_to_records: Convert to list of dictionaries using first row
                as headers.
            numericise_values: Convert string numbers to numeric types.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            range (str): The actual range that was read.
            values (list): Processed data (2D array or list of dicts).
            major_dimension (str): The major dimension.
            render_options (dict): Applied rendering options.
            processing_options (dict): Applied processing options.

        Raises:
            ValueError: If `spreadsheet_id` or `range_name` is invalid.
            Exception: For API errors or connection issues.
        """
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
    """Input schema for `SheetsBatchReadDataTool`."""

    ranges: List[str] = Field(
        description="List of A1 notation ranges to read from the spreadsheet."
    )

    major_dimension: MajorDimension = Field(
        default=MajorDimension.ROWS,
        description="The major dimension that results should use.",
    )


class SheetsBatchReadDataTool(BaseReadTool):
    """Tool for reading data from multiple ranges in Google Sheets efficiently.

    Inherits from
    [`BaseReadTool`][langchain_google_community.sheets.read_sheet_tools.BaseReadTool].

    Reads multiple ranges in a single API call, reducing network overhead and
    improving performance.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        requested_ranges (list): The ranges that were requested.
        total_ranges (int): Total number of ranges processed.
        successful_ranges (int): Number of successfully processed ranges.
        failed_ranges (int): Number of failed ranges.
        results (list): List of results for each range with data and error fields.
        value_render_option (str): Applied value rendering option.
        date_time_render_option (str): Applied date/time rendering option.
        major_dimension (str): Applied major dimension.
        convert_to_records (bool): Whether data was converted to records.
        numericise_values (bool): Whether values were numericised.

    ???+ example "Basic Usage"

        Read multiple ranges in one call:

        ```python
        from langchain_google_community.sheets import SheetsBatchReadDataTool

        tool = SheetsBatchReadDataTool(api_key="your_api_key")
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "ranges": ["A1:C5", "F1:H5", "Sheet2!A1:D10"],
            }
        )
        for r in result["results"]:
            print(f"{r['range']}: {len(r['data'])} rows")
        ```

    ??? example "With Record Conversion"

        Read and convert to dictionaries:

        ```python
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "ranges": ["Sheet1!A1:D10", "Sheet2!A1:E10"],
                "convert_to_records": True,
            }
        )
        ```

    Raises:
        ValueError: If `spreadsheet_id` is invalid or ranges list is empty.
        Exception: For API errors or connection issues.
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
        """Read data from multiple ranges in a Google Spreadsheet efficiently.

        Args:
            spreadsheet_id: ID of the spreadsheet to read from.
            ranges: List of A1 notation ranges to read.
            value_render_option: How values should be rendered.
            date_time_render_option: How dates/times should be rendered.
            major_dimension: Major dimension for results.
            convert_to_records: Convert to list of dictionaries.
            numericise_values: Convert string numbers to numeric types.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            requested_ranges (list): The ranges that were requested.
            total_ranges (int): Total number of ranges processed.
            successful_ranges (int): Number of successfully processed ranges.
            failed_ranges (int): Number of failed ranges.
            results (list): List of results for each range.
            value_render_option (str): Applied value rendering option.
            date_time_render_option (str): Applied date/time rendering option.
            major_dimension (str): Applied major dimension.
            convert_to_records (bool): Whether data was converted to records.
            numericise_values (bool): Whether values were numericised.

        Raises:
            ValueError: If `spreadsheet_id` is invalid or ranges list is empty.
            Exception: For API errors or connection issues.
        """
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
    """Schema for `DataFilter` used with `getByDataFilter` API."""

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
    """Input schema for `SheetsFilteredReadDataTool`."""

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
    """Tool for reading data from Google Sheets using DataFilters.

    [`BaseReadTool`][langchain_google_community.sheets.read_sheet_tools.BaseReadTool].

    Uses `getByDataFilter` API to read ranges specified by A1 notation, grid
    coordinates, or developer metadata. Optionally includes detailed cell formatting.

    !!! note "Authentication Required"

        Requires OAuth2 authentication (not API key).

    !!! warning "Range Selection Only"

        This tool is for `RANGE SELECTION`, not conditional filtering like
        `'score > 50'`. For conditional filtering, create a Filter View using
        Sheets UI or `batchUpdate` API.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): The spreadsheet ID.
        properties (dict): Spreadsheet-level properties.
        sheets (list): List of sheets with filtered data containing properties
            and data segments.

    ???+ example "Basic Usage with A1 Notation"

        Read using A1 notation:

        ```python
        from langchain_google_community.sheets import SheetsFilteredReadDataTool

        tool = SheetsFilteredReadDataTool(api_resource=service)
        result = tool.run(
            {
                "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "data_filters": [{"a1Range": "Sheet1!A1:E10"}],
                "include_grid_data": True,
            }
        )
        ```

    ??? example "Using Grid Coordinates"

        Read using grid coordinates:

        ```python
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
            }
        )
        ```

    Raises:
        ValueError: If `spreadsheet_id` is invalid or `data_filters` is empty.
        Exception: For API errors, authentication issues, or connection problems.
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
        """Read data from Google Sheets using DataFilters.

        Args:
            spreadsheet_id: ID of the spreadsheet to read from.
            data_filters: List of DataFilter objects specifying ranges to read.
            include_grid_data: Include detailed cell formatting.
            value_render_option: How values should be rendered.
            date_time_render_option: How dates/times should be rendered.
            convert_to_records: Convert to list of dictionaries.
            numericise_values: Convert string numbers to numeric types.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): The spreadsheet ID.
            properties (dict): Spreadsheet-level properties.
            sheets (list): List of sheets with filtered data.

        Raises:
            ValueError: If `spreadsheet_id` is invalid or `data_filters` is empty.
            Exception: For API errors, authentication issues, or connection problems.
        """
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
