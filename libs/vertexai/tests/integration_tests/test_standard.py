"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_vertexai import ChatVertexAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


@pytest.mark.first
class TestGemini2AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-2.0-flash-exp",
            "rate_limiter": rate_limiter,
            "temperature": 0,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True


@pytest.mark.first
class TestGeminiAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-1.0-pro-001",
            "rate_limiter": rate_limiter,
            "temperature": 0,
        }

    @pytest.mark.xfail(reason="Gemini 1.0 doesn't support tool_choice='any'")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        super().test_structured_few_shot_examples(model)


class TestGemini_15_AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-1.5-pro-001",
            "rate_limiter": rate_limiter,
            "temperature": 0,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True
