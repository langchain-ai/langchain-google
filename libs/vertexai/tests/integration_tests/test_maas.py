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

from langchain_google_vertexai.model_garden_maas.mistral import (
    VertexModelGardenMistral,
)

model_names = [
    "mistral-nemo@2407",
    "mistral-large@2407",
]


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_generate(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_agenerate(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = await llm.ainvoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_stream(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.stream("What is the meaning of life?")
    for chunk in output:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_astream(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.astream("What is the meaning of life?")
    async for chunk in output:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
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

    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
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
    assert len(tool_calls) == 1

    tool_response = search("sparrow")
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
    assert "brown" in result.content
    assert len(result.tool_calls) == 0
