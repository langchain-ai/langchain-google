"""Test for protobuf integer/float conversion fix in chat models."""

import json

from google.ai.generativelanguage_v1beta.types import (
    Candidate,
    Content,
    FunctionCall,
    Part,
)
from google.protobuf.struct_pb2 import Struct

from langchain_google_genai.chat_models import _parse_response_candidate


def test_parse_response_candidate_corrects_integer_like_floats() -> None:
    """Test that _parse_response_candidate correctly handles integer-like floats.

    Handling in tool call arguments from the Gemini API response.

    This test addresses a bug where proto.Message.to_dict() converts integers
    to floats, causing downstream type casting errors.
    """
    # Create a mock Protobuf Struct for the arguments with problematic float values
    args_struct = Struct()
    args_struct.update(
        {
            "entity_type": "table",
            "upstream_depth": 3.0,  # The problematic float value that should be int
            "downstream_depth": 5.0,  # Another problematic float value
            "fqn": "test.table.name",
            "valid_float": 3.14,  # This should remain as float
            "string_param": "test_string",  # This should remain as string
            "bool_param": True,  # This should remain as boolean
        }
    )

    # Create the mock API response candidate
    candidate = Candidate(
        content=Content(
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="get_entity_lineage",
                        args=args_struct,
                    )
                )
            ]
        )
    )

    # Call the function we are testing
    result_message = _parse_response_candidate(candidate)

    # Assert that the parsed tool_calls have the correct integer types
    assert len(result_message.tool_calls) == 1
    tool_call = result_message.tool_calls[0]
    assert tool_call["name"] == "get_entity_lineage"
    assert tool_call["args"]["upstream_depth"] == 3
    assert tool_call["args"]["downstream_depth"] == 5
    assert isinstance(tool_call["args"]["upstream_depth"], int)
    assert isinstance(tool_call["args"]["downstream_depth"], int)

    # Assert that non-integer values are preserved correctly
    assert tool_call["args"]["valid_float"] == 3.14
    assert isinstance(tool_call["args"]["valid_float"], float)
    assert tool_call["args"]["string_param"] == "test_string"
    assert isinstance(tool_call["args"]["string_param"], str)
    assert tool_call["args"]["bool_param"] is True
    assert isinstance(tool_call["args"]["bool_param"], bool)

    # Assert that the additional_kwargs also contains corrected JSON
    function_call_args = json.loads(
        result_message.additional_kwargs["function_call"]["arguments"]
    )
    assert function_call_args["upstream_depth"] == 3
    assert function_call_args["downstream_depth"] == 5
    assert isinstance(function_call_args["upstream_depth"], int)
    assert isinstance(function_call_args["downstream_depth"], int)

    # Assert that non-integer values are preserved in additional_kwargs too
    assert function_call_args["valid_float"] == 3.14
    assert isinstance(function_call_args["valid_float"], float)


def test_parse_response_candidate_handles_no_function_call() -> None:
    """Test that the function works correctly when there's no function call."""
    candidate = Candidate(
        content=Content(
            parts=[Part(text="This is a regular text response without function calls")]
        )
    )

    result_message = _parse_response_candidate(candidate)

    assert (
        result_message.content
        == "This is a regular text response without function calls"
    )
    assert len(result_message.tool_calls) == 0
    assert "function_call" not in result_message.additional_kwargs


def test_parse_response_candidate_handles_empty_args() -> None:
    """Test that the function works correctly with empty function call arguments."""
    args_struct = Struct()
    # Empty struct - no arguments

    candidate = Candidate(
        content=Content(
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="no_args_function",
                        args=args_struct,
                    )
                )
            ]
        )
    )

    result_message = _parse_response_candidate(candidate)

    assert len(result_message.tool_calls) == 1
    tool_call = result_message.tool_calls[0]
    assert tool_call["name"] == "no_args_function"
    assert tool_call["args"] == {}

    function_call_args = json.loads(
        result_message.additional_kwargs["function_call"]["arguments"]
    )
    assert function_call_args == {}
