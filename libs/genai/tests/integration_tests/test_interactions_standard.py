"""Standard LangChain interface tests for Gemini Interactions API."""

import os
from typing import Literal

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

pytestmark = pytest.mark.flaky(retries=3, delay=1)

rate_limiter = InMemoryRateLimiter(requests_per_second=2.0)

_FLASH_MODEL = "gemini-3.8-flash"
_FLASH_OVERRIDE_MODEL = "gemini-3.6-flash"
_ARGON_MODEL = "gemini-4-argon"
_VERTEX_INTERACTIONS_LOCATIONS = frozenset({"global", "us", "eu"})


def _get_vertex_interactions_location() -> str:
    """Return a Vertex AI location supported by the Interactions API.

    Vertex Interactions rejects regional locations such as `us-central1`, so
    default to `'global'` unless `GOOGLE_CLOUD_LOCATION` is explicitly one of
    the supported endpoints (`'global'`, `'us'`, `'eu'`).
    """
    loc = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip().lower()
    return loc if loc in _VERTEX_INTERACTIONS_LOCATIONS else "global"


def _has_multimodal_secrets() -> bool:
    """Check if integration test secrets are available.

    Returns `True` if running in an environment with access to secrets.
    """
    return bool(os.environ.get("LANGCHAIN_TESTS_USER_AGENT"))


def _get_backend_configs() -> list[tuple[str, dict]]:
    """Get list of backend configurations based on `TEST_VERTEXAI` env var.

    Returns:
        List of `(backend_name, backend_config)` tuples.
    """
    vertexai_setting = os.environ.get("TEST_VERTEXAI", "").lower()

    configs: list[tuple[str, dict]] = []

    if vertexai_setting != "only":
        configs.append(("GoogleAI", {}))

    if vertexai_setting in ("1", "true", "yes", "only"):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = _get_vertex_interactions_location()
        configs.append(
            (
                "VertexAI",
                {
                    "vertexai": True,
                    "project": project,
                    "location": location,
                    "api_key": None,
                }
                if project
                else {
                    "vertexai": True,
                    "project": None,
                    "location": location,
                    "api_key": None,
                },
            )
        )

    return configs


def _create_standard_test_classes(
    backend_name: str, backend_config: dict
) -> tuple[type[ChatModelIntegrationTests], type[ChatModelIntegrationTests]]:
    """Create backend-bound standard test classes without loop-closure capture."""
    bound_config = dict(backend_config)

    class _TestGeminiInteractionsFlashStandardBase(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[BaseChatModel]:
            return ChatGoogleGenerativeAI

        @property
        def chat_model_params(self) -> dict:
            if bound_config.get("vertexai") and not bound_config.get("project"):
                pytest.skip(
                    "Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set"
                )
            return {
                "model": _FLASH_MODEL,
                "use_interactions_api": True,
                "rate_limiter": rate_limiter,
                "timeout": 120,
                **bound_config,
            }

        @property
        def model_override_value(self) -> str:
            return _FLASH_OVERRIDE_MODEL

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
        def supports_pdf_tool_message(self) -> bool:
            return True

        @property
        def supports_audio_inputs(self) -> bool:
            return True

        @property
        def supports_anthropic_inputs(self) -> bool:
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

    @pytest.mark.long
    class _TestGeminiInteractionsArgonStandardBase(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[BaseChatModel]:
            return ChatGoogleGenerativeAI

        @property
        def chat_model_params(self) -> dict:
            if bound_config.get("vertexai") and not bound_config.get("project"):
                pytest.skip(
                    "Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set"
                )
            return {
                "model": _ARGON_MODEL,
                "use_interactions_api": True,
                "rate_limiter": rate_limiter,
                "timeout": 180,
                **bound_config,
            }

        @property
        def model_override_value(self) -> str:
            return _FLASH_MODEL

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
        def supports_pdf_tool_message(self) -> bool:
            return True

        @property
        def supports_audio_inputs(self) -> bool:
            return True

        @property
        def supports_anthropic_inputs(self) -> bool:
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

    flash_class_name = f"TestGeminiInteractionsFlashStandard{backend_name}"
    argon_class_name = f"TestGeminiInteractionsArgonStandard{backend_name}"

    flash_cls = type(flash_class_name, (_TestGeminiInteractionsFlashStandardBase,), {})
    argon_cls = type(argon_class_name, (_TestGeminiInteractionsArgonStandardBase,), {})
    return flash_cls, argon_cls


for _backend_name, _backend_config in _get_backend_configs():
    _flash_cls, _argon_cls = _create_standard_test_classes(
        _backend_name, _backend_config
    )
    globals()[_flash_cls.__name__] = _flash_cls
    globals()[_argon_cls.__name__] = _argon_cls
