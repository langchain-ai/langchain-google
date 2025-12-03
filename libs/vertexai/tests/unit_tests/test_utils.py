import asyncio
from typing import Any, NoReturn
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import ClientError, GoogleAPICallError, InvalidArgument

from langchain_google_vertexai._retry import create_base_retry_decorator
from langchain_google_vertexai._utils import (
    _get_def_key_from_schema_path,
    _strip_nullable_anyof,
    get_generation_info,
    get_user_agent,
    replace_defs_in_schema,
)


def test_valid_schema_path() -> None:
    schema_path = "#/$defs/MyDefinition"
    expected_key = "MyDefinition"
    assert _get_def_key_from_schema_path(schema_path) == expected_key


@pytest.mark.parametrize(
    "schema_path",
    [123, "#/definitions/MyDefinition", "#/$defs/MyDefinition/extra", "#/$defs"],
)
def test_invalid_schema_path(schema_path: Any) -> None:
    with pytest.raises(ValueError):
        _get_def_key_from_schema_path(schema_path)


def test_schema_no_defs() -> None:
    schema = {"type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_empty_defs() -> None:
    schema = {"$defs": {}, "type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_simple_ref_replacement() -> None:
    schema = {
        "$defs": {"MyDefinition": {"type": "string"}},
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {"property": {"type": "string"}}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_nested_ref_replacement() -> None:
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


def test_schema_recursive_error_self_reference() -> None:
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


def test_retry_decorator_for_google_api_call_error_and_subclass() -> None:
    google_api_call_error_retries = []
    client_error_retries = []
    max_retries = 3

    retry_decorator = create_base_retry_decorator([GoogleAPICallError], max_retries)

    @retry_decorator
    def retry_for_google_api_call_error() -> bool:
        google_api_call_error_retries.append("retried")
        if len(google_api_call_error_retries) == max_retries:
            # This method executes successfully in the last retry
            return True

        msg = ""
        raise GoogleAPICallError(msg)

    @retry_decorator
    def retry_for_subclass_of_google_api_call_error() -> bool:
        client_error_retries.append("retried")
        if len(client_error_retries) == max_retries:
            # This method executes successfully in the last retry
            return True

        msg = ""
        raise ClientError(msg)

    google_api_call_error_retried = retry_for_google_api_call_error()
    client_error_retried = retry_for_subclass_of_google_api_call_error()

    assert google_api_call_error_retried
    assert client_error_retried
    assert len(google_api_call_error_retries) == max_retries
    assert len(client_error_retries) == max_retries


def test_retry_decorator_for_invalid_argument() -> None:
    invalid_argument_retries = []
    max_retries = 3

    retry_decorator = create_base_retry_decorator([GoogleAPICallError], max_retries)

    @retry_decorator
    def retry_for_invalid_argument_error() -> NoReturn:
        invalid_argument_retries.append("retried")
        msg = ""
        raise InvalidArgument(msg)

    try:
        retry_for_invalid_argument_error()
    except InvalidArgument:
        # Silently handling the raised exception
        pass

    assert len(invalid_argument_retries) == 1


@patch("langchain_google_vertexai._utils.os.environ.get")
@patch("langchain_google_vertexai._utils._LANGCHAIN_VERTEXAI_VERSION", "1.2.3")
def test_get_user_agent_with_telemetry_env_variable(
    mock_environ_get: MagicMock,
) -> None:
    mock_environ_get.return_value = True
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module+remote_reasoning_engine"
    assert user_agent_str == (
        "langchain-google-vertexai/1.2.3-test-module+remote_reasoning_engine"
    )


@patch("langchain_google_vertexai._utils.os.environ.get")
@patch("langchain_google_vertexai._utils._LANGCHAIN_VERTEXAI_VERSION", "1.2.3")
def test_get_user_agent_without_telemetry_env_variable(
    mock_environ_get: MagicMock,
) -> None:
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


def test_get_generation_info_with_raw_int_finish_reason() -> None:
    """Test that get_generation_info handles raw integer finish_reason values."""
    # Create a mock candidate with finish_reason as raw int (e.g., 15)
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = 15  # Raw integer value
    mock_candidate.finish_message = None
    mock_candidate.safety_ratings = []
    mock_candidate.citation_metadata = None
    mock_candidate.grounding_metadata = None

    # get_generation_info should not crash and should handle the raw int
    result = get_generation_info(mock_candidate)

    assert "finish_reason" in result
    assert result["finish_reason"] == "UNKNOWN_15"


def test_get_generation_info_with_enum_finish_reason() -> None:
    """Test that get_generation_info handles normal enum finish_reason values."""
    # Create a mock candidate with finish_reason as enum with .name attribute
    mock_finish_reason = MagicMock()
    mock_finish_reason.name = "STOP"

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = mock_finish_reason
    mock_candidate.finish_message = None
    mock_candidate.safety_ratings = []
    mock_candidate.citation_metadata = None
    mock_candidate.grounding_metadata = None

    result = get_generation_info(mock_candidate)

    assert "finish_reason" in result
    assert result["finish_reason"] == "STOP"


def test_get_generation_info_with_none_finish_reason() -> None:
    """Test that get_generation_info handles None finish_reason values."""
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = None
    mock_candidate.finish_message = None
    mock_candidate.safety_ratings = []
    mock_candidate.citation_metadata = None
    mock_candidate.grounding_metadata = None

    result = get_generation_info(mock_candidate)

    # (Should be None for None finish_reason)
    assert result.get("finish_reason") is None


def test_version_is_cached_at_module_level() -> None:
    """Test that version is cached at module level and doesn't call metadata.version."""
    from langchain_google_vertexai import _utils

    # The cached version should be a string
    assert isinstance(_utils._LANGCHAIN_VERTEXAI_VERSION, str)
    # Should be either a valid version or "0.0.0" (fallback)
    assert _utils._LANGCHAIN_VERTEXAI_VERSION != ""


async def test_get_user_agent_no_blocking_in_async_context() -> None:
    """Test that get_user_agent doesn't perform blocking I/O in async context.

    This test verifies that get_user_agent uses the cached version
    and doesn't call metadata.version() which would be blocking I/O.
    """
    # Mock metadata.version to raise an error if it's called
    with patch("langchain_google_vertexai._utils.metadata.version") as mock_version:
        mock_version.side_effect = RuntimeError(
            "metadata.version() should not be called - version should be cached"
        )

        # This should work without calling metadata.version()
        client_lib_version, user_agent_str = get_user_agent(module="test-async")

        # Verify the call didn't happen
        mock_version.assert_not_called()

        # Verify we got valid output (using the cached version)
        assert "test-async" in client_lib_version
        assert "langchain-google-vertexai" in user_agent_str


def test_async_context_execution() -> None:
    """Run the async test to ensure it works in event loop."""
    asyncio.run(test_get_user_agent_no_blocking_in_async_context())


def test_get_generation_info_logprobs_with_zero_values() -> None:
    """Test that zero logprobs are included, not filtered out."""
    mock_chosen_zero = MagicMock()
    mock_chosen_zero.token = "certain_token"
    mock_chosen_zero.log_probability = 0.0

    mock_chosen_negative = MagicMock()
    mock_chosen_negative.token = "probable_token"
    mock_chosen_negative.log_probability = -0.5

    mock_chosen_int = MagicMock()
    mock_chosen_int.token = "int_token"
    mock_chosen_int.log_probability = 0  # Integer zero (also valid)

    # Create mock top candidates for each chosen candidate
    mock_top_zero = MagicMock()
    mock_top_zero.token = "top_certain"
    mock_top_zero.log_probability = 0.0

    mock_top_negative = MagicMock()
    mock_top_negative.token = "top_probable"
    mock_top_negative.log_probability = -1.2

    # Create top_candidates structure
    mock_top_candidates_0 = MagicMock()
    mock_top_candidates_0.candidates = [mock_top_zero, mock_top_negative]

    mock_top_candidates_1 = MagicMock()
    mock_top_candidates_1.candidates = [mock_top_negative]

    mock_top_candidates_2 = MagicMock()
    mock_top_candidates_2.candidates = [mock_top_zero]

    # Create mock logprobs_result
    mock_logprobs_result = MagicMock()
    mock_logprobs_result.chosen_candidates = [
        mock_chosen_zero,
        mock_chosen_negative,
        mock_chosen_int,
    ]
    mock_logprobs_result.top_candidates = [
        mock_top_candidates_0,
        mock_top_candidates_1,
        mock_top_candidates_2,
    ]

    # Create mock candidate
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = None
    mock_candidate.finish_message = None
    mock_candidate.safety_ratings = []
    mock_candidate.citation_metadata = None
    mock_candidate.grounding_metadata = None
    mock_candidate.logprobs_result = mock_logprobs_result

    # Call get_generation_info with logprobs=2 to also get top_logprobs
    result = get_generation_info(mock_candidate, logprobs=2)

    assert "logprobs_result" in result
    logprobs_result = result["logprobs_result"]

    # Should have all 3 tokens (including zero logprobs)
    assert len(logprobs_result) == 3

    assert logprobs_result[0]["token"] == "certain_token"
    assert logprobs_result[0]["logprob"] == 0.0

    assert logprobs_result[1]["token"] == "probable_token"
    assert logprobs_result[1]["logprob"] == -0.5

    assert logprobs_result[2]["token"] == "int_token"
    assert logprobs_result[2]["logprob"] == 0

    assert len(logprobs_result[0]["top_logprobs"]) == 2
    assert logprobs_result[0]["top_logprobs"][0]["logprob"] == 0.0


def test_get_generation_info_logprobs_filters_invalid_values() -> None:
    """Test that invalid logprob values (positive, NaN) are filtered out."""

    # Create mock chosen candidates with invalid values
    mock_chosen_positive = MagicMock()
    mock_chosen_positive.token = "invalid_positive"
    mock_chosen_positive.log_probability = 0.5  # Invalid: positive

    mock_chosen_nan = MagicMock()
    mock_chosen_nan.token = "invalid_nan"
    mock_chosen_nan.log_probability = float("nan")  # Invalid: NaN

    mock_chosen_valid = MagicMock()
    mock_chosen_valid.token = "valid"
    mock_chosen_valid.log_probability = -0.3

    # Create mock top_candidates structure
    mock_top_candidates = MagicMock()
    mock_top_candidates.candidates = []

    mock_logprobs_result = MagicMock()
    mock_logprobs_result.chosen_candidates = [
        mock_chosen_positive,
        mock_chosen_nan,
        mock_chosen_valid,
    ]
    mock_logprobs_result.top_candidates = [
        mock_top_candidates,
        mock_top_candidates,
        mock_top_candidates,
    ]

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = None
    mock_candidate.finish_message = None
    mock_candidate.safety_ratings = []
    mock_candidate.citation_metadata = None
    mock_candidate.grounding_metadata = None
    mock_candidate.logprobs_result = mock_logprobs_result

    result = get_generation_info(mock_candidate, logprobs=True)

    # Should only have 1 valid token (positive and NaN filtered out)
    assert "logprobs_result" in result
    assert len(result["logprobs_result"]) == 1
    assert result["logprobs_result"][0]["token"] == "valid"
    assert result["logprobs_result"][0]["logprob"] == -0.3
