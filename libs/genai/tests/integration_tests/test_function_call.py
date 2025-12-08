"""Test `ChatGoogleGenerativeAI` function calling abilities."""

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
)

MODEL_NAMES = ["gemini-2.5-flash-lite"]


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_function_call(model_name: str, backend_config: dict) -> None:
    functions = [
        {
            "name": "get_weather",
            "description": "Determine weather in my location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location"],
            },
        }
    ]
    llm = ChatGoogleGenerativeAI(model=model_name, **backend_config).bind(
        functions=functions, tool_choice="any"
    )
    res = llm.invoke("what weather is today in san francisco?")
    assert res
    assert res.additional_kwargs
    assert "function_call" in res.additional_kwargs
    assert res.additional_kwargs["function_call"]["name"] == "get_weather"
    arguments_str = res.additional_kwargs["function_call"]["arguments"]
    assert isinstance(arguments_str, str)
    arguments = json.loads(arguments_str)
    assert "location" in arguments


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_tool_call(model_name: str, backend_config: dict) -> None:
    @tool
    def search_tool(query: str) -> str:
        """Searches the web for `query` and returns the result."""
        raise NotImplementedError

    llm = ChatGoogleGenerativeAI(model=model_name, **backend_config).bind(
        functions=[search_tool], tool_choice="any"
    )
    response = llm.invoke("weather in san francisco")
    assert isinstance(response, AIMessage)
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "search_tool"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert "query" in arguments


class MyModel(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_pydantic_call(model_name: str, backend_config: dict) -> None:
    llm = ChatGoogleGenerativeAI(model=model_name, **backend_config).bind(
        functions=[MyModel], tool_choice="any"
    )
    response = llm.invoke("my name is Erick and I am 27 years old")
    assert isinstance(response, AIMessage)
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "MyModel"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert arguments == {
        "name": "Erick",
        "age": 27.0,
    }
