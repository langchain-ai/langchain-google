import json
from typing import Dict
from unittest.mock import ANY

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_google_vertexai.model_garden import VertexMaaS, _parse_response_candidate


def test_llama_init() -> None:
    llm = VertexMaaS(
        model="meta/llama3-405b-instruct-maas",
        location="moon-dark",
        project="test-project",
    )
    assert llm._llm_type == "vertexai_model_garden_maas"
    assert llm.model_name == "meta/llama3-405b-instruct-maas"

    assert (
        llm.get_url()
        == "https://moon-dark-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/moon-dark/endpoints/openapi/chat/completions"
    )
    assert (
        llm.get_url(stream=True)
        == "https://moon-dark-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/moon-dark/endpoints/openapi/chat/completions"
    )


def test_mistral_init() -> None:
    llm = VertexMaaS(
        model="mistral-large@2407", location="us-central1", project="test-project"
    )
    assert llm._llm_type == "vertexai_model_garden_maas"
    assert llm.model_name == "mistral-large"
    assert llm.full_model_name == "mistral-large@2407"
    assert (
        llm.get_url()
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/mistralai/models/mistral-large:rawPredict"
    )
    assert (
        llm.get_url(stream=True)
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/mistralai/models/mistral-large:streamRawPredict"
    )


def test_parse_history() -> None:
    llm = VertexMaaS(
        model="meta/llama3-405b-instruct-maas",
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


def test_parse_history_llama_tools() -> None:
    @tool
    def get_weather(city: str) -> float:
        """Get the current weather and temperature for a given city."""
        return 23.0

    schema = convert_to_openai_function(get_weather)

    llm = VertexMaaS(
        model="meta/llama3-405b-instruct-maas",
        location="us-central1",
        project="test-project",
        append_tools_to_system_message=True,
    )
    history = [
        SystemMessage(content="You're a helpful assistant."),
        HumanMessage(content="What is the weather in Munich?"),
    ]
    parsed_history = llm._convert_messages(history, tools=[get_weather])  # type: ignore[list-item]
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
    parsed_history = llm._convert_messages(history, tools=[get_weather])  # type: ignore[list-item]
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


def test_parse_response():
    candidate: Dict[str, str] = {
        "content": "London is the capital of Great Britain",
        "role": "assistant",
    }
    assert _parse_response_candidate(candidate) == AIMessage(
        content="London is the capital of Great Britain"
    )
    candidate = {
        "content": ('{"name": "test_tool", "parameters": {"arg1": "test", "arg2": 2}}'),
        "role": "assistant",
    }
    parsed = _parse_response_candidate(candidate)
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
