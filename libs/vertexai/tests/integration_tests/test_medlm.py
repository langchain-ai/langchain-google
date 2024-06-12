"""Test MedlLM models.
- medlm-large & medlm-medium are PALM family, should return str.
- medlm-medium@latest is part of GEMINI family,
    - should return str for VertexAI/Text Completion,
    - should returnAIMessage for ChatVertexAI/Chat Completion"""

import pytest
from langchain_core.messages import (
    AIMessage,
)

from langchain_google_vertexai import ChatVertexAI, VertexAI
from tests.integration_tests.conftest import _DEFAULT_MODEL_NAME

model_names_to_test = [None, "codechat-bison", "chat-bison", _DEFAULT_MODEL_NAME]


@pytest.mark.extended
def test_invoke_medlm_large() -> None:
    model = VertexAI(model_name="medlm-large")
    result = model.invoke("How you can help me?")
    assert isinstance(result, str)


@pytest.mark.extended
def test_invoke_medlm_large_error() -> None:
    with pytest.raises(ValueError):
        model = ChatVertexAI(model_name="medlm-large")
        model.invoke("How you can help me?")


@pytest.mark.extended
def test_invoke_medlm_medium() -> None:
    model = VertexAI(model_name="medlm-medium")
    result = model.invoke("How you can help me?")
    assert isinstance(result, str)


@pytest.mark.extended
def test_invoke_medlm_medium_error() -> None:
    with pytest.raises(ValueError):
        model = ChatVertexAI(model_name="medlm-medium")
        model.invoke("How you can help me?")


@pytest.mark.extended
def test_invoke_test_completion() -> None:
    model = VertexAI(model_name="medlm-medium@latest")
    result = model.invoke("How you can help me?")
    assert isinstance(result, str)


@pytest.mark.extended
def test_invoke_chat() -> None:
    model = ChatVertexAI(model_name="medlm-medium@latest")
    result = model.invoke("How you can help me?")
    assert isinstance(result, AIMessage)
