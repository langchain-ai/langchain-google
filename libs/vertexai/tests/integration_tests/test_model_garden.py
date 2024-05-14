import os
from typing import Optional

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import LLMResult

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


@pytest.mark.xfail(reason="CI issue")
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


@pytest.mark.xfail(reason="CI issue")
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


@pytest.mark.xfail(reason="CI issue")
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
