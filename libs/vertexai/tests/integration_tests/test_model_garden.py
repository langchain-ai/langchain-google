import json
import os
from typing import Optional

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_vertexai.model_garden import (
    ChatAnthropicVertex,
    VertexAIModelGarden,
)


@pytest.mark.extended
@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT_ID=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm("What is the meaning of life?")
    assert isinstance(output, str)
    print(output)
    assert llm._llm_type == "vertexai_model_garden"


@pytest.mark.extended
@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden_generate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export FALCON_ENDPOINT_ID=...
    export LLAMA_ENDPOINT_ID=...
    export PROJECT_ID=...
    """
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = llm.generate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.extended
@pytest.mark.asyncio
@pytest.mark.first
@pytest.mark.parametrize(
    "endpoint_os_variable_name,result_arg",
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
async def test_model_garden_agenerate(
    endpoint_os_variable_name: str, result_arg: Optional[str]
) -> None:
    endpoint_id = os.environ[endpoint_os_variable_name]
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    llm = VertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        result_arg=result_arg,
        location=location,
    )
    output = await llm.agenerate(["What is the meaning of life?", "How much is 2+2"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 2


@pytest.mark.extended
def test_anthropic() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    raw_context = (
        "My name is Peter. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model.invoke([context, message], model_name="claude-3-sonnet@20240229")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.extended
def test_anthropic_stream() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message = HumanMessage(content=question)
    sync_response = model.stream([message], model="claude-3-sonnet@20240229")
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
def test_anthropic_thinking_stream() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-east5"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
        model_kwargs={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,  # budget tokens >= 1024
            },
        },
        max_tokens=2048,  # max_tokens must be greater than budget_tokens
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message = HumanMessage(content=question)
    sync_response = model.stream([message], model="claude-3-7-sonnet@20250219")
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
async def test_anthropic_async() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    raw_context = (
        "My name is Peter. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = await model.ainvoke(
        [context, message], model_name="claude-3-sonnet@20240229", temperature=0.2
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def _check_tool_calls(response: BaseMessage, expected_name: str) -> None:
    """Check tool calls are as expected."""
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    tool_calls = response.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == expected_name
    assert tool_call["args"] == {"age": 27.0, "name": "Erick"}


@pytest.mark.extended
@pytest.mark.flaky(retries=3)
def test_anthropic_tool_calling() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-east5"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    class MyModel(BaseModel):
        name: str
        age: int

    # Test .bind_tools with BaseModel
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    model_with_tools = model.bind_tools([MyModel], model_name="claude-3-opus@20240229")
    response = model_with_tools.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int) -> None:
        """Invoke this with names and ages."""
        pass

    model_with_tools = model.bind_tools([my_model], model_name="claude-3-opus@20240229")
    response = model_with_tools.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool
    @tool
    def my_tool(name: str, age: int) -> None:
        """Invoke this with names and ages."""
        pass

    model_with_tools = model.bind_tools([my_tool], model_name="claude-3-opus@20240229")
    response = model_with_tools.invoke([message])
    _check_tool_calls(response, "my_tool")

    # Test streaming
    stream = model_with_tools.stream([message])
    first = True
    for chunk in stream:
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "my_tool"
    assert tool_call_chunk["args"]
    if tool_call_chunk["args"]:
        assert json.loads(tool_call_chunk["args"]) == {"age": 27.0, "name": "Erick"}


@pytest.mark.extended
def test_anthropic_with_structured_output() -> None:
    project = os.environ["PROJECT_ID"]
    location = "us-east5"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
        model="claude-3-opus@20240229",
    )

    class MyModel(BaseModel):
        name: str
        age: int

    message = HumanMessage(content="My name is Erick and I am 27 years old")
    model_with_structured_output = model.with_structured_output(MyModel)
    response = model_with_structured_output.invoke([message])

    assert isinstance(response, MyModel)
    assert response.name == "Erick"
    assert response.age == 27
