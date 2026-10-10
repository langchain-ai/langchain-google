"""Integration tests for Google Sheets tools."""

import os

import pytest

from langchain_google_community.sheets import (
    SheetsBatchReadDataTool,
    SheetsGetSpreadsheetInfoTool,
    SheetsReadDataTool,
    SheetsToolkit,
)

# Substrings that mark a Google Sheets API error as "unavailable in this
# environment" rather than a genuine tool failure: the API is not enabled on
# the project (`SERVICE_DISABLED` / "API has not been used"), or the API key is
# restricted from calling it (`API_KEY_SERVICE_BLOCKED` / "are blocked", as
# happens in the langchain-google CI project). When matched, the affected
# integration test is skipped instead of failed.
_SHEETS_API_UNAVAILABLE_MARKERS = (
    "SERVICE_DISABLED",
    "API has not been used",
    "API_KEY_SERVICE_BLOCKED",
    "are blocked",
)


def _is_sheets_api_unavailable(error: Exception) -> bool:
    """Return whether the error means the Sheets API is unavailable in CI."""
    message = str(error)
    return any(marker in message for marker in _SHEETS_API_UNAVAILABLE_MARKERS)


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

        # Verify dict response (tools now return dicts, not JSON strings)
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert "range" in result
        assert "values" in result
        assert len(result["values"]) > 0
        assert "A1:C3" in result["range"]
    except Exception as e:
        if _is_sheets_api_unavailable(e):
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

        # Verify dict response (tools now return dicts, not JSON strings)
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert "spreadsheet_id" in result
        assert "results" in result
        assert len(result["results"]) == 2
    except Exception as e:
        if _is_sheets_api_unavailable(e):
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

        # Verify dict response (tools now return dicts, not JSON strings)
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert "spreadsheet_id" in result
        assert "title" in result
        assert "sheets" in result
        assert len(result["sheets"]) > 0
    except Exception as e:
        if _is_sheets_api_unavailable(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise


@pytest.mark.extended
def test_toolkit_functionality(sheets_api_key: str, test_spreadsheet_id: str) -> None:
    """Test SheetsToolkit functionality with API key (read-only)."""
    toolkit = SheetsToolkit(api_key=sheets_api_key)
    tools = toolkit.get_tools()

    # Should have 3 read-only tools (no write tools with API key)
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

        # Verify dict response (tools now return dicts, not JSON strings)
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
        assert "range" in result
        assert "values" in result
        assert "A1:B2" in result["range"]
    except Exception as e:
        if _is_sheets_api_unavailable(e):
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
        if _is_sheets_api_unavailable(e):
            pytest.skip("Google Sheets API not enabled in CI environment")
        raise
