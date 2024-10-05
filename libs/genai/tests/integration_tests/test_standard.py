"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)


class TestGeminiAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-1.5-pro-001",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="Not yet supported")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)


# TODO: increase quota on gemini-1.5-pro-001 and test as well
