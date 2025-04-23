import json
from typing import List

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_google_vertexai.model_garden_maas import (
    _LLAMA_MODELS,
    _MISTRAL_MODELS,
    get_vertex_maas_model,
)

model_names = _LLAMA_MODELS + _MISTRAL_MODELS
# Fix tool support for new Mistral and Llama models
model_names_with_tools_support = [
    "mistral-nemo@2407",
]


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_generate(model_name: str) -> None:
    llm = get_vertex_maas_model(model_name=model_name, location="us-central1")
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_agenerate(model_name: str) -> None:
    llm = get_vertex_maas_model(model_name=model_name, location="us-central1")
    output = await llm.ainvoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_stream(model_name: str) -> None:
    # streaming currently fails with mistral-nemo@2407
    if "stral" in model_name:
        return
    llm = get_vertex_maas_model(model_name=model_name, location="us-central1")
    output = llm.stream("What is the meaning of life?")
    for chunk in output:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_astream(model_name: str) -> None:
    # streaming currently fails with mistral-nemo@2407
    if "stral" in model_name:
        return
    llm = get_vertex_maas_model(model_name=model_name, location="us-central1")
    output = llm.astream("What is the meaning of life?")
    async for chunk in output:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names_with_tools_support)
@pytest.mark.flaky(retries=3)
async def test_tools(model_name: str) -> None:
    @tool
    def search(
        question: str,
    ) -> str:
        """
        Useful for when you need to answer questions or visit websites.
        You should ask targeted questions.
        """
        return "brown"

    tools = [search]

    llm = get_vertex_maas_model(
        model_name=model_name,
        location="us-central1",
        append_tools_to_system_message=True,
    )
    llm_with_search = llm.bind_tools(
        tools=tools,
    )
    llm_with_search_force = llm_with_search.bind(
        tool_choice={"type": "function", "function": {"name": "search"}}
    )
    request = HumanMessage(
        content="Please tell the primary color of sparrow?",
    )
    response = llm_with_search_force.invoke([request])

    assert isinstance(response, AIMessage)
    tool_calls = response.tool_calls
    assert len(tool_calls) > 0

    tool_response = search.invoke("sparrow")
    tool_messages: List[BaseMessage] = []

    for tool_call in tool_calls:
        assert tool_call["name"] == "search"
        tool_message = ToolMessage(
            name=tool_call["name"],
            content=json.dumps(tool_response),
            tool_call_id=(tool_call["id"] or ""),
        )
        tool_messages.append(tool_message)

    result = llm_with_search.invoke([request, response] + tool_messages)

    assert isinstance(result, AIMessage)
    if model_name in _MISTRAL_MODELS:
        assert "brown" in result.content
    assert len(result.tool_calls) == 0
