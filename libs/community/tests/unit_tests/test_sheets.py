"""Unit tests for Google Sheets tools."""

from unittest.mock import MagicMock

from langchain_google_community.sheets import (
    SheetsBatchReadDataTool,
    SheetsGetSpreadsheetInfoTool,
    SheetsReadDataTool,
    SheetsToolkit,
)


def test_sheets_read_data_tool() -> None:
    """Test SheetsReadDataTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsReadDataTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_read_data"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "range_name" in schema_fields


def test_sheets_batch_read_data_tool() -> None:
    """Test SheetsBatchReadDataTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsBatchReadDataTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_batch_read_data"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "ranges" in schema_fields


def test_sheets_get_spreadsheet_info_tool() -> None:
    """Test SheetsGetSpreadsheetInfoTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsGetSpreadsheetInfoTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_get_spreadsheet_info"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields


def test_sheets_toolkit_with_api_key() -> None:
    """Test SheetsToolkit with API key authentication."""
    toolkit = SheetsToolkit(api_key="test_api_key")
    tools = toolkit.get_tools()

    # Should return 3 tools (excludes filtered read which requires OAuth2)
    assert len(tools) == 3
    tool_names = [tool.name for tool in tools]
    assert "sheets_read_data" in tool_names
    assert "sheets_batch_read_data" in tool_names
    assert "sheets_get_spreadsheet_info" in tool_names
    assert "sheets_filtered_read_data" not in tool_names


def test_sheets_toolkit_with_oauth2() -> None:
    """Test SheetsToolkit with OAuth2 authentication."""
    # Test that toolkit can be instantiated with OAuth2
    toolkit = SheetsToolkit()

    # Test that the toolkit has the correct attributes
    assert toolkit.api_resource is None  # Default value
    assert toolkit.api_key is None  # Default value

    # Test that the toolkit can be configured
    toolkit.api_key = "test_key"
    assert toolkit.api_key == "test_key"
