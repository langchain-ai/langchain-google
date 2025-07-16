from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import ClientError, GoogleAPICallError, InvalidArgument

from langchain_google_vertexai._retry import create_base_retry_decorator
from langchain_google_vertexai._utils import (
    _get_def_key_from_schema_path,
    _strip_nullable_anyof,
    get_user_agent,
    replace_defs_in_schema,
)


def test_valid_schema_path():
    schema_path = "#/$defs/MyDefinition"
    expected_key = "MyDefinition"
    assert _get_def_key_from_schema_path(schema_path) == expected_key


@pytest.mark.parametrize(
    "schema_path",
    [123, "#/definitions/MyDefinition", "#/$defs/MyDefinition/extra", "#/$defs"],
)
def test_invalid_schema_path(schema_path: Any):
    with pytest.raises(ValueError):
        _get_def_key_from_schema_path(schema_path)


def test_schema_no_defs():
    schema = {"type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_empty_defs():
    schema = {"$defs": {}, "type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_simple_ref_replacement():
    schema = {
        "$defs": {"MyDefinition": {"type": "string"}},
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {"property": {"type": "string"}}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_nested_ref_replacement():
    schema = {
        "$defs": {
            "MyDefinition": {
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/NameDefinition"}},
            },
            "NameDefinition": {"type": "string"},
        },
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {
        "property": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_recursive_error_self_reference():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }
        },
        "root": {"$ref": "#/$defs/Node"},
    }
    with pytest.raises(RecursionError):
        _ = replace_defs_in_schema(schema)


def test_retry_decorator_for_google_api_call_error_and_subclass():
    google_api_call_error_retries = []
    client_error_retries = []
    max_retries = 3

    retry_decorator = create_base_retry_decorator([GoogleAPICallError], max_retries)

    @retry_decorator
    def retry_for_google_api_call_error():
        google_api_call_error_retries.append("retried")
        if len(google_api_call_error_retries) == max_retries:
            # This method executes successfully in the last retry
            return True

        raise GoogleAPICallError("")

    @retry_decorator
    def retry_for_subclass_of_google_api_call_error():
        client_error_retries.append("retried")
        if len(client_error_retries) == max_retries:
            # This method executes successfully in the last retry
            return True

        raise ClientError("")

    google_api_call_error_retried = retry_for_google_api_call_error()
    client_error_retried = retry_for_subclass_of_google_api_call_error()

    assert google_api_call_error_retried
    assert client_error_retried
    assert len(google_api_call_error_retries) == max_retries
    assert len(client_error_retries) == max_retries


def test_retry_decorator_for_invalid_argument():
    invalid_argument_retries = []
    max_retries = 3

    retry_decorator = create_base_retry_decorator([GoogleAPICallError], max_retries)

    @retry_decorator
    def retry_for_invalid_argument_error():
        invalid_argument_retries.append("retried")
        raise InvalidArgument("")

    try:
        retry_for_invalid_argument_error()
    except InvalidArgument:
        # Silently handling the raised exception
        pass

    assert len(invalid_argument_retries) == 1


@patch("langchain_google_vertexai._utils.os.environ.get")
@patch("langchain_google_vertexai._utils.metadata.version")
def test_get_user_agent_with_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = True
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module+remote_reasoning_engine"
    assert user_agent_str == (
        "langchain-google-vertexai/1.2.3-test-module+remote_reasoning_engine"
    )


@patch("langchain_google_vertexai._utils.os.environ.get")
@patch("langchain_google_vertexai._utils.metadata.version")
def test_get_user_agent_without_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = False
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module"
    assert user_agent_str == "langchain-google-vertexai/1.2.3-test-module"


def test_strip_nullable_anyof() -> None:
    input_schema = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {
                "anyOf": [
                    {"type": "null"},
                    {"type": "integer"},
                ]
            },
            "field3": {"type": "boolean"},
        },
        "required": ["field1", "field2", "field3"],
    }
    expected = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"},
            "field3": {"type": "boolean"},
        },
        "required": ["field1", "field3"],
    }
    assert _strip_nullable_anyof(input_schema) == expected

    # Nested schemas
    input_schema = {
        "properties": {
            "fruits": {
                "items": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "color": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Color",
                        },
                    },
                    "required": ["name", "color"],
                    "title": "Fruit",
                    "type": "object",
                },
                "title": "Fruits",
                "type": "array",
            }
        },
        "required": ["fruits"],
        "title": "FruitList",
        "type": "object",
    }

    expected = {
        "properties": {
            "fruits": {
                "items": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "color": {"title": "Color", "type": "string"},
                    },
                    "required": ["name"],
                    "title": "Fruit",
                    "type": "object",
                },
                "title": "Fruits",
                "type": "array",
            }
        },
        "required": ["fruits"],
        "title": "FruitList",
        "type": "object",
    }
    assert _strip_nullable_anyof(input_schema) == expected
