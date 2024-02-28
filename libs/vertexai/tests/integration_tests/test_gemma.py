import os

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from langchain_google_vertexai import (
    GemmaChatLocalKaggle,
    GemmaChatVertexAIModelGarden,
    GemmaLocalKaggle,
    GemmaVertexAIModelGarden,
)


@pytest.mark.extended
def test_gemma_model_garden() -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export GEMMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ["GEMMA_ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    location = "us-central1"
    llm = GemmaVertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        location=location,
    )
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, str)
    assert len(output) > 2
    assert llm._llm_type == "gemma_vertexai_model_garden"


@pytest.mark.extended
def test_gemma_chat_model_garden() -> None:
    """In order to run this test, you should provide endpoint names.

    Example:
    export GEMMA_ENDPOINT_ID=...
    export PROJECT=...
    """
    endpoint_id = os.environ["GEMMA_ENDPOINT_ID"]
    project = os.environ["PROJECT"]
    location = "us-central1"
    llm = GemmaChatVertexAIModelGarden(
        endpoint_id=endpoint_id,
        project=project,
        location=location,
    )
    assert llm._llm_type == "gemma_vertexai_model_garden"

    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    output = llm.invoke([message1])
    assert isinstance(output, AIMessage)
    assert len(output.content) > 2
    output = llm.invoke([message1, message2, message3])
    assert isinstance(output, AIMessage)
    assert len(output.content) > 2


@pytest.mark.gpu
def test_gemma_kaggle() -> None:
    llm = GemmaLocalKaggle(model_name="gemma_2b_en")
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, str)
    print(output)
    assert len(output) > 2


@pytest.mark.gpu
def test_gemma_chat_kaggle() -> None:
    llm = GemmaChatLocalKaggle(model_name="gemma_2b_en")
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    output = llm.invoke([message1])
    assert isinstance(output, AIMessage)
    assert len(output.content) > 2
    output = llm.invoke([message1, message2, message3])
    assert isinstance(output, AIMessage)
    assert len(output.content) > 2
