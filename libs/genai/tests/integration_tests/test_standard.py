"""Standard LangChain interface tests."""

import os
from typing import Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)

_FLASH_MODEL = "gemini-2.5-flash"
_PRO_MODEL = "gemini-3-pro-preview"


def _has_multimodal_secrets() -> bool:
    """Check if integration test secrets are available.

    Returns `True` if running in an environment with access to secrets.
    """
    return bool(os.environ.get("LANGCHAIN_TESTS_USER_AGENT"))


def _get_backend_configs() -> list[tuple[str, dict]]:
    """Get list of backend configurations based on TEST_VERTEXAI env var.

    Returns:
        List of (backend_name, backend_config) tuples.
    """
    vertexai_setting = os.environ.get("TEST_VERTEXAI", "").lower()

    configs: list = []

    # Add Google AI config
    if vertexai_setting != "only":
        configs.append(("GoogleAI", {}))

    # Add Vertex AI config
    if vertexai_setting in ("1", "true", "yes", "only"):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        # Always add config, tests will skip if project is missing
        configs.append(
            (
                "VertexAI",
                {"vertexai": True, "project": project, "api_key": None}
                if project
                else {"vertexai": True, "project": None, "api_key": None},
            )
        )

    return configs


# Dynamically create test classes for each backend
for backend_name, backend_config in _get_backend_configs():

    class _TestGeminiFlashStandardBase(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[BaseChatModel]:
            return ChatGoogleGenerativeAI

        @property
        def chat_model_params(self) -> dict:
            # Skip if Vertex AI is requested but GOOGLE_CLOUD_PROJECT is not set
            if backend_config.get("vertexai") and not backend_config.get("project"):
                pytest.skip(
                    "Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set"
                )
            return {
                "model": _FLASH_MODEL,
                "rate_limiter": rate_limiter,
                **backend_config,
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

    class _TestGeminiProStandardBase(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[BaseChatModel]:
            return ChatGoogleGenerativeAI

        @property
        def chat_model_params(self) -> dict:
            # Skip if Vertex AI is requested but GOOGLE_CLOUD_PROJECT is not set
            if backend_config.get("vertexai") and not backend_config.get("project"):
                pytest.skip(
                    "Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set"
                )
            return {
                "model": _PRO_MODEL,
                "rate_limiter": rate_limiter,
                **backend_config,
            }

        @property
        def supports_image_inputs(self) -> bool:
            return True

        @property
        def supports_image_urls(self) -> bool:
            return True

        @property
        def supports_pdf_inputs(self) -> bool:
            return True

        @property
        def supports_audio_inputs(self) -> bool:
            return True

        @property
        def supports_json_mode(self) -> bool:
            return True

        @property
        def supports_image_tool_message(self) -> bool:
            return True

        @property
        def supports_pdf_tool_message(self) -> bool:
            return True

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

    # Create backend-specific test classes
    flash_class_name = f"TestGeminiFlashStandard{backend_name}"
    pro_class_name = f"TestGeminiProStandard{backend_name}"

    # Add classes to module globals so pytest can discover them
    globals()[flash_class_name] = type(
        flash_class_name, (_TestGeminiFlashStandardBase,), {}
    )
    globals()[pro_class_name] = type(pro_class_name, (_TestGeminiProStandardBase,), {})
