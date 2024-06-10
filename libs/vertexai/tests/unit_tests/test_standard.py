from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_google_vertexai import ChatVertexAI


class TestGeminiAIStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {"model_name": "gemini-1.0-pro-001"}


class TestGemini_15_AIStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {"model_name": "gemini-1.5-pro-001"}
