import json
from typing import Any
from unittest.mock import ANY, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_google_vertexai.model_garden_maas import get_vertex_maas_model
from langchain_google_vertexai.model_garden_maas.llama import (
    _parse_response_candidate_llama,
)

_MODEL_NAME = "meta/llama-3.3-70b-instruct-maas"


@patch("langchain_google_vertexai.model_garden_maas._base.auth")
def test_llama_init(mock_auth: Any) -> None:
    mock_credentials = MagicMock()
    mock_credentials.token.return_value = "test-token"
    mock_auth.default.return_value = (mock_credentials, None)
    llm = get_vertex_maas_model(
        model_name=_MODEL_NAME,
        location="moon-dark",
        project="test-project",
    )
    assert llm._llm_type == "vertexai_model_garden_maas_llama"
    assert llm.model_name == _MODEL_NAME

    assert (
        llm.get_url()
        == "https://moon-dark-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/moon-dark"
    )
    assert llm._get_url_part() == "endpoints/openapi/chat/completions"
    assert llm._get_url_part(stream=True) == "endpoints/openapi/chat/completions"
    mock_credentials.refresh.assert_called_once()


@patch("langchain_google_vertexai.model_garden_maas._base.auth")
def test_parse_history(mock_auth: Any) -> None:
    llm = get_vertex_maas_model(
        model_name=_MODEL_NAME,
        location="us-central1",
        project="test-project",
    )
    history = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="What is the capital of Great Britain?"),
        AIMessage(content="London is a capital of Great Britain"),
    ]
    parsed_history = llm._convert_messages(history)
    expected_parsed_history = [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "What is the capital of Great Britain?"},
        {"role": "assistant", "content": "London is a capital of Great Britain"},
    ]
    assert parsed_history == expected_parsed_history


@patch("langchain_google_vertexai.model_garden_maas._base.auth")
def test_parse_history_llama_tools(mock_auth: Any) -> None:
    @tool
    def get_weather(city: str) -> float:
        """Get the current weather and temperature for a given city."""
        return 23.0

    schema = convert_to_openai_function(get_weather)

    llm = get_vertex_maas_model(
        model_name=_MODEL_NAME,
        location="us-central1",
        project="test-project",
        append_tools_to_system_message=True,
    )
    history = [
        SystemMessage(content="You're a helpful assistant."),
        HumanMessage(content="What is the weather in Munich?"),
    ]
    parsed_history = llm._convert_messages(history, tools=[get_weather])
    expected_parsed_history = [
        {"role": "system", "content": ANY},
        {"role": "user", "content": "What is the weather in Munich?"},
    ]
    assert parsed_history == expected_parsed_history
    assert json.dumps(schema) in parsed_history[0]["content"]

    history += [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "Munich"},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="32", name="get_weather", tool_call_id="1"),
    ]
    parsed_history = llm._convert_messages(history, tools=[get_weather])
    expected_parsed_history = [
        {"role": "system", "content": ANY},
        {"role": "user", "content": "What is the weather in Munich?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Munich"}',
                    },
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "32", "tool_call_id": "1"},
    ]
    assert parsed_history == expected_parsed_history


def test_parse_response() -> None:
    candidate: dict[str, Any] = {
        "content": "London is the capital of Great Britain",
        "role": "assistant",
    }
    assert _parse_response_candidate_llama(candidate) == AIMessage(
        content="London is the capital of Great Britain"
    )
    candidate = {
        "content": ('{"name": "test_tool", "parameters": {"arg1": "test", "arg2": 2}}'),
        "role": "assistant",
    }
    parsed = _parse_response_candidate_llama(candidate)
    assert isinstance(parsed, AIMessage)
    assert parsed.content == ""
    assert parsed.tool_calls == [
        {
            "name": "test_tool",
            "args": {"arg1": "test", "arg2": 2},
            "id": ANY,
            "type": "tool_call",
        }
    ]
    candidate = {
        "tool_calls": [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": '{"arg1": "test", "arg2": 2}',
                }
            }
        ],
        "role": "assistant",
    }
    parsed = _parse_response_candidate_llama(candidate)
    assert isinstance(parsed, AIMessage)
    assert parsed.content == ""
    assert parsed.tool_calls == [
        {
            "name": "test_tool",
            "args": {"arg1": "test", "arg2": 2},
            "id": ANY,
            "type": "tool_call",
        }
    ]
