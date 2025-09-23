"""Integration tests for Google Sheets tools."""

import os

import pytest

from langchain_google_community.sheets import (
    SheetsBatchReadDataTool,
    SheetsGetSpreadsheetInfoTool,
    SheetsReadDataTool,
    SheetsToolkit,
)


@pytest.fixture
def sheets_api_key() -> str:
    """Get Google Sheets API key from environment."""
    api_key = os.getenv("GOOGLE_SHEETS_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_SHEETS_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def test_spreadsheet_id() -> str:
    """Get test spreadsheet ID from environment."""
    spreadsheet_id = os.getenv("TEST_SPREADSHEET_ID")
    if not spreadsheet_id:
        pytest.skip("TEST_SPREADSHEET_ID environment variable not set")
    return spreadsheet_id


@pytest.fixture
def sheets_read_tool(sheets_api_key: str) -> SheetsReadDataTool:
    """Create SheetsReadDataTool with API key."""
    return SheetsReadDataTool(api_key=sheets_api_key)


@pytest.fixture
def sheets_batch_tool(sheets_api_key: str) -> SheetsBatchReadDataTool:
    """Create SheetsBatchReadDataTool with API key."""
    return SheetsBatchReadDataTool(api_key=sheets_api_key)


@pytest.fixture
def sheets_info_tool(sheets_api_key: str) -> SheetsGetSpreadsheetInfoTool:
    """Create SheetsGetSpreadsheetInfoTool with API key."""
    return SheetsGetSpreadsheetInfoTool(api_key=sheets_api_key)


@pytest.fixture
def sheets_toolkit(sheets_api_key: str) -> SheetsToolkit:
    """Create SheetsToolkit with API key."""
    return SheetsToolkit(api_key=sheets_api_key)


@pytest.mark.extended
def test_read_sheet_data(
    sheets_read_tool: SheetsReadDataTool, test_spreadsheet_id: str
) -> None:
    """Test reading data from a Google Sheet."""
    result = sheets_read_tool.run(
        {"spreadsheet_id": test_spreadsheet_id, "range_name": "A1:C3"}
    )

    # Parse the JSON result
    import json

    data = json.loads(result)

    # Verify the response structure
    assert "spreadsheet_id" in data
    assert "range_name" in data
    assert "values" in data
    assert "records" in data
    assert "metadata" in data

    # Verify we got some data
    assert len(data["values"]) > 0
    assert data["spreadsheet_id"] == test_spreadsheet_id
    assert data["range_name"] == "A1:C3"


@pytest.mark.extended
def test_batch_read_sheet_data(
    sheets_batch_tool: SheetsBatchReadDataTool, test_spreadsheet_id: str
) -> None:
    """Test batch reading data from multiple ranges."""
    result = sheets_batch_tool.run(
        {"spreadsheet_id": test_spreadsheet_id, "ranges": ["A1:C3", "D1:F3"]}
    )

    # Parse the JSON result
    import json

    data = json.loads(result)

    # Verify the response structure
    assert "spreadsheet_id" in data
    assert "ranges" in data
    assert "results" in data
    assert "metadata" in data

    # Verify we got results for both ranges
    assert len(data["ranges"]) == 2
    assert len(data["results"]) == 2
    assert "A1:C3" in data["results"]
    assert "D1:F3" in data["results"]


@pytest.mark.extended
def test_get_spreadsheet_info(
    sheets_info_tool: SheetsGetSpreadsheetInfoTool, test_spreadsheet_id: str
) -> None:
    """Test getting spreadsheet metadata."""
    result = sheets_info_tool.run(
        {"spreadsheet_id": test_spreadsheet_id, "fields": "properties,sheets"}
    )

    # Parse the JSON result
    import json

    data = json.loads(result)

    # Verify the response structure
    assert "spreadsheet_id" in data
    assert "properties" in data
    assert "sheets" in data

    # Verify we got spreadsheet properties
    assert "title" in data["properties"]
    assert len(data["sheets"]) > 0


@pytest.mark.extended
def test_toolkit_functionality(
    sheets_toolkit: SheetsToolkit, test_spreadsheet_id: str
) -> None:
    """Test SheetsToolkit functionality."""
    tools = sheets_toolkit.get_tools()

    # Should return 3 tools (excludes filtered read which requires OAuth2)
    assert len(tools) == 3
    tool_names = [tool.name for tool in tools]
    assert "sheets_read_data" in tool_names
    assert "sheets_batch_read_data" in tool_names
    assert "sheets_get_spreadsheet_info" in tool_names

    # Test that we can use a tool from the toolkit
    read_tool = next(tool for tool in tools if tool.name == "sheets_read_data")
    result = read_tool.run(
        {"spreadsheet_id": test_spreadsheet_id, "range_name": "A1:B2"}
    )

    # Verify the result
    import json

    data = json.loads(result)
    assert data["spreadsheet_id"] == test_spreadsheet_id
    assert data["range_name"] == "A1:B2"


@pytest.mark.extended
def test_error_handling_invalid_range(
    sheets_read_tool: SheetsReadDataTool, test_spreadsheet_id: str
) -> None:
    """Test error handling with invalid range."""
    result = sheets_read_tool.run(
        {"spreadsheet_id": test_spreadsheet_id, "range_name": "InvalidRange!A1:Z999"}
    )

    # Parse the JSON result
    import json

    data = json.loads(result)

    # Should contain error information
    assert "error" in data or "status" in data
