"""Unit tests for Google Sheets tools."""

from unittest.mock import MagicMock

from langchain_google_community.sheets import (
    SheetsAppendValuesTool,
    SheetsBatchReadDataTool,
    SheetsBatchUpdateValuesTool,
    SheetsClearValuesTool,
    SheetsCreateSpreadsheetTool,
    SheetsFilteredReadDataTool,
    SheetsGetSpreadsheetInfoTool,
    SheetsReadDataTool,
    SheetsToolkit,
    SheetsUpdateValuesTool,
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


def test_sheets_filtered_read_data_tool() -> None:
    """Test SheetsFilteredReadDataTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsFilteredReadDataTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_get_by_data_filter"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "data_filters" in schema_fields


def test_sheets_create_spreadsheet_tool() -> None:
    """Test SheetsCreateSpreadsheetTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsCreateSpreadsheetTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_create_spreadsheet"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "title" in schema_fields


def test_sheets_update_values_tool() -> None:
    """Test SheetsUpdateValuesTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsUpdateValuesTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_update_values"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "range" in schema_fields
    assert "values" in schema_fields


def test_sheets_append_values_tool() -> None:
    """Test SheetsAppendValuesTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsAppendValuesTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_append_values"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "range" in schema_fields
    assert "values" in schema_fields


def test_sheets_clear_values_tool() -> None:
    """Test SheetsClearValuesTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsClearValuesTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_clear_values"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "range" in schema_fields


def test_sheets_batch_update_values_tool() -> None:
    """Test SheetsBatchUpdateValuesTool basic functionality."""
    mock_api_resource = MagicMock()
    tool = SheetsBatchUpdateValuesTool.model_construct(api_resource=mock_api_resource)

    # Test that the tool has the correct schema
    assert tool.args_schema is not None
    assert tool.name == "sheets_batch_update_values"

    # Test that required fields are present
    schema_fields = tool.args_schema.model_fields
    assert "spreadsheet_id" in schema_fields
    assert "data" in schema_fields


def test_sheets_toolkit_with_api_key() -> None:
    """Test SheetsToolkit with API key (read-only)."""
    toolkit = SheetsToolkit(api_key="test_api_key")
    tools = toolkit.get_tools()

    # Should return 3 read-only tools (no write tools with API key)
    assert len(tools) == 3
    tool_names = [tool.name for tool in tools]

    # Read-only tools should be present
    assert "sheets_read_data" in tool_names
    assert "sheets_batch_read_data" in tool_names
    assert "sheets_get_spreadsheet_info" in tool_names

    # OAuth2-only tools should NOT be present
    assert "sheets_get_by_data_filter" not in tool_names
    assert "sheets_create_spreadsheet" not in tool_names
    assert "sheets_update_values" not in tool_names
    assert "sheets_append_values" not in tool_names
    assert "sheets_clear_values" not in tool_names
    assert "sheets_batch_update_values" not in tool_names


def test_sheets_toolkit_with_oauth2() -> None:
    """Test SheetsToolkit with OAuth2 (full access)."""
    # Test that toolkit can be instantiated with OAuth2
    toolkit = SheetsToolkit()

    # Test that the toolkit has the correct attributes
    assert toolkit.api_resource is None  # Default value
    assert toolkit.api_key is None  # Default value

    # Test that the toolkit can be configured
    toolkit.api_key = "test_key"
    assert toolkit.api_key == "test_key"
