import json
import os

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_vertexai.model_garden import (
    ChatAnthropicVertex,
    VertexAIModelGarden,
)

_ANTHROPIC_LOCATION = "us-east5"
_ANTHROPIC_CLAUDE_MODEL_NAME = "claude-sonnet-4-5@20250929"


@pytest.mark.extended
@pytest.mark.parametrize(
    ("endpoint_os_variable_name", "result_arg"),
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden(endpoint_os_variable_name: str, result_arg: str | None) -> None:
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
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, str)
    assert llm._llm_type == "vertexai_model_garden"


@pytest.mark.extended
@pytest.mark.parametrize(
    ("endpoint_os_variable_name", "result_arg"),
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
def test_model_garden_generate(
    endpoint_os_variable_name: str, result_arg: str | None
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
    ("endpoint_os_variable_name", "result_arg"),
    [("FALCON_ENDPOINT_ID", "generated_text"), ("LLAMA_ENDPOINT_ID", None)],
)
async def test_model_garden_agenerate(
    endpoint_os_variable_name: str, result_arg: str | None
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
    location = _ANTHROPIC_LOCATION
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
    response = model.invoke([context, message], model_name=_ANTHROPIC_CLAUDE_MODEL_NAME)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.extended
def test_anthropic_stream() -> None:
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message = HumanMessage(content=question)
    sync_response = model.stream([message], model=_ANTHROPIC_CLAUDE_MODEL_NAME)
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
    sync_response = model.stream([message], model="claude-sonnet-4-5@20250929")
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
async def test_anthropic_async() -> None:
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
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
        [context, message], model_name=_ANTHROPIC_CLAUDE_MODEL_NAME, temperature=0.2
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
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    class MyModel(BaseModel):
        name: str
        age: int

    # Test .bind_tools with BaseModel
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    model_with_tools = model.bind_tools(
        [MyModel], model_name=_ANTHROPIC_CLAUDE_MODEL_NAME
    )
    response = model_with_tools.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int) -> None:
        """Invoke this with names and ages."""

    model_with_tools = model.bind_tools(
        [my_model], model_name=_ANTHROPIC_CLAUDE_MODEL_NAME
    )
    response = model_with_tools.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool
    @tool
    def my_tool(name: str, age: int) -> None:
        """Invoke this with names and ages."""

    model_with_tools = model.bind_tools(
        [my_tool], model_name=_ANTHROPIC_CLAUDE_MODEL_NAME
    )
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
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
        model=_ANTHROPIC_CLAUDE_MODEL_NAME,
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


@pytest.mark.extended
@pytest.mark.flaky(retries=3)
def test_anthropic_multiturn_tool_calling() -> None:
    """Test multi-turn conversation with tool calls and responses.

    This test ensures that ToolMessages with streaming metadata are properly
    cleaned before being sent back to the API (fixes issue #1227).
    """
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
        model=_ANTHROPIC_CLAUDE_MODEL_NAME,
    )

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Sunny, 22°C in {city}"

    # Bind tools to model
    model_with_tools = model.bind_tools([get_weather])

    # First turn - user asks question, model calls tool
    user_message = HumanMessage("What's the weather in Paris?")
    response1 = model_with_tools.invoke([user_message])

    # Verify model made a tool call
    assert isinstance(response1, AIMessage)
    assert response1.tool_calls
    assert len(response1.tool_calls) == 1
    assert response1.tool_calls[0]["name"] == "get_weather"
    assert response1.tool_calls[0]["args"]["city"] == "Paris"

    # Second turn - provide tool result (this is where the bug was)
    # The ToolMessage might contain streaming metadata that needs cleaning
    tool_result = ToolMessage(
        content="Sunny, 22°C in Paris", tool_call_id=response1.tool_calls[0]["id"]
    )

    # This should NOT raise "Extra inputs are not permitted" error
    response2 = model.invoke([user_message, response1, tool_result])

    # Verify model responded with final answer
    assert isinstance(response2, AIMessage)
    assert isinstance(response2.content, str)
    assert "paris" in response2.content.lower()


@pytest.mark.extended
@pytest.mark.flaky(retries=3)
def test_anthropic_tool_error_handling() -> None:
    """Test that tool errors are properly communicated with is_error flag."""
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
        model=_ANTHROPIC_CLAUDE_MODEL_NAME,
    )

    @tool
    def failing_tool(x: int) -> str:
        """A tool that simulates failure."""
        return "result"

    model_with_tools = model.bind_tools([failing_tool])

    # First turn - model calls tool
    response1 = model_with_tools.invoke([HumanMessage("Use failing_tool with x=5")])

    # Verify tool call
    assert isinstance(response1, AIMessage)
    assert response1.tool_calls
    assert response1.tool_calls[0]["name"] == "failing_tool"

    # Second turn - simulate tool error with status="error"
    error_result = ToolMessage(
        content="Tool execution failed: API error",
        tool_call_id=response1.tool_calls[0]["id"],
        status="error",
    )

    # Should handle error gracefully (is_error flag should be sent)
    response2 = model.invoke(
        [HumanMessage("Use failing_tool with x=5"), response1, error_result]
    )

    # Verify model acknowledged the error
    assert isinstance(response2, AIMessage)
    assert isinstance(response2.content, str)


@pytest.mark.extended
async def test_anthropic_async_invoke_with_betas() -> None:
    """Test that async invoke method handles betas parameter correctly."""
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    message = HumanMessage(content="What is 2+2?")

    # This should work but currently fails because _agenerate doesn't handle betas
    response = await model.ainvoke(
        [message],
        model_name=_ANTHROPIC_CLAUDE_MODEL_NAME,
        betas=["context-1m-2025-08-07"],
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.extended
async def test_anthropic_async_stream_with_betas() -> None:
    """Test that async streaming handles betas parameter correctly."""
    project = os.environ["PROJECT_ID"]
    location = _ANTHROPIC_LOCATION
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )
    message = HumanMessage(content="Say hello")

    # This should work but currently fails because _astream doesn't handle betas
    chunks = []
    async for chunk in model.astream(
        [message],
        model=_ANTHROPIC_CLAUDE_MODEL_NAME,
        betas=["context-1m-2025-08-07"],
    ):
        chunks.append(chunk)
        assert isinstance(chunk, AIMessageChunk)

    assert len(chunks) > 0
