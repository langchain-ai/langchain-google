from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import ClientError, GoogleAPICallError, InvalidArgument

from langchain_google_vertexai._retry import create_base_retry_decorator
from langchain_google_vertexai._utils import (
    GoogleModelFamily,
    _get_def_key_from_schema_path,
    get_user_agent,
    replace_defs_in_schema,
)


@pytest.mark.parametrize(
    "srcs,exp",
    [
        (
            [
                "chat-bison@001",
                "text-bison@002",
                "medlm-medium",
                "medlm-large",
            ],
            GoogleModelFamily.PALM,
        ),
        (
            [
                "code-bison@002",
                "code-gecko@002",
            ],
            GoogleModelFamily.CODEY,
        ),
        (
            [
                "gemini-1.0-pro-001",
                "gemini-1.0-pro-002",
                "gemini-1.0-pro-vision-001",
                "gemini-1.0-pro-vision",
                "medlm-medium@latest",
            ],
            GoogleModelFamily.GEMINI,
        ),
        (
            [
                "gemini-1.5-flash-preview-0514",
                "gemini-1.5-pro-preview-0514",
                "gemini-1.5-pro-preview-0409",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-001",
                "medlm-large-1.5-preview",
                "medlm-large-1.5-001",
            ],
            GoogleModelFamily.GEMINI_ADVANCED,
        ),
    ],
)
def test_google_model_family(srcs: List[str], exp: GoogleModelFamily):
    for src in srcs:
        res = GoogleModelFamily(src)
        assert res == exp


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
