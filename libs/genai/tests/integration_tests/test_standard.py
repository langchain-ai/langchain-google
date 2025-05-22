"""Standard LangChain interface tests"""

from typing import Dict, List, Literal, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)


class TestGeminiAI2Standard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-2.0-flash-001",
            "rate_limiter": rate_limiter,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason="Likely a bug in genai: prompt_token_count inconsistent in final chunk."
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason="investigate")
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)

    @pytest.mark.xfail(reason=("investigate"))
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)


class TestGeminiAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-1.5-pro-latest",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(reason="Not yet supported")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @pytest.mark.xfail(
        reason="Investigate: prompt_token_count inconsistent in final chunk."
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {"invoke": [], "stream": []}
