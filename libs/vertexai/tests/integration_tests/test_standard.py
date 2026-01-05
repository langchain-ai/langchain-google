"""Standard LangChain interface tests."""

import os

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_vertexai import ChatVertexAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


def _has_multimodal_secrets() -> bool:
    """Check if integration test secrets are available.

    Returns `True` if running in an environment with access to secrets.
    """
    return bool(os.environ.get("LANGCHAIN_TESTS_USER_AGENT"))


@pytest.mark.first
class TestGemini2AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-2.5-flash",
            "rate_limiter": rate_limiter,
            "temperature": 0,
            "api_transport": None,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        # TODO: 403 Client Error: Forbidden for this specific URL
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
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


class TestGemini_15_AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-2.5-pro",
            "rate_limiter": rate_limiter,
            "temperature": 0,
            "api_transport": None,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        # TODO: 403 Client Error: Forbidden for this specific URL
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
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
