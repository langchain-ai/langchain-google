"""Test MedlLM models.
- medlm-medium@latest is part of GEMINI family,
    - should return str for VertexAI/Text Completion,
    - should returnAIMessage for ChatVertexAI/Chat Completion"""

import pytest

from langchain_google_vertexai import VertexAI


@pytest.mark.extended
def test_invoke_medlm_large_palm_error() -> None:
    with pytest.raises(ValueError):
        model = VertexAI(model_name="medlm-large")
        model.invoke("How you can help me?")
