"""Test MedlLM models."""

import pytest
from langchain_core.messages import (
    AIMessage,
)

from langchain_google_vertexai import ChatVertexAI, VertexAI
from tests.integration_tests.conftest import _DEFAULT_MODEL_NAME

model_names_to_test = [None, "codechat-bison", "chat-bison", _DEFAULT_MODEL_NAME]


@pytest.mark.extended
def test_invoke() -> None:
    model = VertexAI(model_name="medlm-large")
    result = model.invoke("How you can help me?")
    assert isinstance(result, str)


@pytest.mark.extended
def test_invoke_chat() -> None:
    model = ChatVertexAI(model_name="medlm-medium@latest")
    result = model.invoke("How you can help me?")
    assert isinstance(result, AIMessage)
