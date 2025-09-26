"""Integration tests for Google Sheets tools."""

import json
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def test_spreadsheet_id() -> str:
    """Get test spreadsheet ID - using public Google Sheets sample."""
    return "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"


@pytest.mark.extended
def test_read_sheet_data(sheets_api_key: str, test_spreadsheet_id: str) -> None:
    """Test reading data from a Google Sheet."""
    tool = SheetsReadDataTool.from_api_key(sheets_api_key)

    try:
        result = tool.run(
            {
                "spreadsheet_id": test_spreadsheet_id,
                "range_name": "A1:C3",
                "convert_to_records": True,
            }
        )

        # Parse and verify JSON response
        data = json.loads(result)
        assert "range" in data
        assert "values" in data
        assert len(data["values"]) > 0
        assert "A1:C3" in data["range"]
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) or "API has not been used" in str(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise


@pytest.mark.extended
def test_batch_read_sheet_data(sheets_api_key: str, test_spreadsheet_id: str) -> None:
    """Test batch reading data from multiple ranges."""
    tool = SheetsBatchReadDataTool.from_api_key(sheets_api_key)

    try:
        result = tool.run(
            {
                "spreadsheet_id": test_spreadsheet_id,
                "ranges": ["A1:C3", "D1:F3"],
                "convert_to_records": True,
            }
        )

        # Parse and verify JSON response
        data = json.loads(result)
        assert "spreadsheet_id" in data
        assert "results" in data
        assert len(data["results"]) == 2
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) or "API has not been used" in str(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise


@pytest.mark.extended
def test_get_spreadsheet_info(sheets_api_key: str, test_spreadsheet_id: str) -> None:
    """Test getting spreadsheet metadata."""
    tool = SheetsGetSpreadsheetInfoTool.from_api_key(sheets_api_key)

    try:
        result = tool.run(
            {
                "spreadsheet_id": test_spreadsheet_id,
                "fields": "properties,sheets",
            }
        )

        # Parse and verify JSON response
        data = json.loads(result)
        assert "spreadsheet_id" in data
        assert "title" in data
        assert "sheets" in data
        assert len(data["sheets"]) > 0
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) or "API has not been used" in str(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise


@pytest.mark.extended
def test_toolkit_functionality(sheets_api_key: str, test_spreadsheet_id: str) -> None:
    """Test SheetsToolkit functionality."""
    toolkit = SheetsToolkit(api_key=sheets_api_key)
    tools = toolkit.get_tools()

    # Should have 3 tools (excludes filtered read which requires OAuth2)
    assert len(tools) == 3

    try:
        # Test using a tool from the toolkit
        read_tool = next(tool for tool in tools if tool.name == "sheets_read_data")
        result = read_tool.run(
            {
                "spreadsheet_id": test_spreadsheet_id,
                "range_name": "A1:B2",
            }
        )

        # Verify the result
        data = json.loads(result)
        assert "range" in data
        assert "values" in data
        assert "A1:B2" in data["range"]
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) or "API has not been used" in str(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise


@pytest.mark.extended
def test_error_handling_invalid_range(
    sheets_api_key: str, test_spreadsheet_id: str
) -> None:
    """Test error handling with invalid range."""
    tool = SheetsReadDataTool.from_api_key(sheets_api_key)

    try:
        with pytest.raises(Exception, match="Error reading sheet data"):
            tool.run(
                {
                    "spreadsheet_id": test_spreadsheet_id,
                    "range_name": "InvalidRange!A1:Z999",
                }
            )
    except Exception as e:
        if "SERVICE_DISABLED" in str(e) or "API has not been used" in str(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise
