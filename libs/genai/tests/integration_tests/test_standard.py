"""Standard LangChain interface tests."""

import os
from typing import Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)


def _has_multimodal_secrets() -> bool:
    """Check if integration test secrets are available.

    Returns `True` if running in an environment with access to secrets.
    """
    return bool(os.environ.get("LANGCHAIN_TESTS_USER_AGENT"))


class TestGeminiFlashStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "gemini-2.5-flash",
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
        not _has_multimodal_secrets(),
        reason=(
            "Multimodal tests require integration secrets (user agent to fetch "
            "external resources)"
        ),
        run=False,
    )
    def test_audio_inputs(self, model: BaseChatModel) -> None:
        """Skip audio tests in PR context - requires external resource fetching."""
        super().test_audio_inputs(model)

    @pytest.mark.xfail(
        not _has_multimodal_secrets(),
        reason=(
            "Multimodal tests require integration secrets (user agent to fetch "
            "external resources)"
        ),
        run=False,
    )
    def test_pdf_inputs(self, model: BaseChatModel) -> None:
        """Skip PDF tests in PR context - requires external resource fetching."""
        super().test_pdf_inputs(model)


class TestGeminiProStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "gemini-2.5-pro",
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
    ) -> dict[
        Literal["invoke", "stream"],
        list[
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
