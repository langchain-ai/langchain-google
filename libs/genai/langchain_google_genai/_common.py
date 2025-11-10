import os
from importlib import metadata
from typing import Any, TypedDict

from google.api_core.gapic_v1.client_info import ClientInfo
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr

from langchain_google_genai._enums import (
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
)

_TELEMETRY_TAG = "remote_reasoning_engine"
_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

# Cache package version at module import time to avoid blocking I/O in async contexts
try:
    LC_GOOGLE_GENAI_VERSION = metadata.version("langchain-google-genai")
except metadata.PackageNotFoundError:
    LC_GOOGLE_GENAI_VERSION = "0.0.0"


class GoogleGenerativeAIError(Exception):
    """Custom exception class for errors associated with the `Google GenAI` API."""


class _BaseGoogleGenerativeAI(BaseModel):
    """Base class for Google Generative AI LLMs."""

    model: str = Field(...)
    """Model name to use."""

    google_api_key: SecretStr | None = Field(
        alias="api_key", default_factory=secret_from_env("GOOGLE_API_KEY", default=None)
    )
    """Google AI API key.

    If not specified will be read from env var `GOOGLE_API_KEY`.
    """

    credentials: Any = None
    """The default custom credentials to use when making API calls.

    If not provided, credentials will be ascertained from the `GOOGLE_API_KEY` env var.
    """

    temperature: float = 0.7
    """Run inference with this temperature.

    Must be within `[0.0, 2.0]`.
    """

    top_p: float | None = None
    """Decode using nucleus sampling.

    Consider the smallest set of tokens whose probability sum is at least `top_p`.

    Must be within `[0.0, 1.0]`.
    """

    top_k: int | None = None
    """Decode using top-k sampling: consider the set of `top_k` most probable tokens.

    Must be positive.
    """

    max_output_tokens: int | None = Field(default=None, alias="max_tokens")
    """Maximum number of tokens to include in a candidate.

    Must be greater than zero.

    If unset, will use the model's default value, which varies by model.

    See [docs](https://ai.google.dev/gemini-api/docs/models) for model-specific limits.
    """

    n: int = 1
    """Number of chat completions to generate for each prompt.

    Note that the API may not return the full `n` completions if duplicates are
    generated.
    """

    max_retries: int = Field(default=6, alias="retries")
    """The maximum number of retries to make when generating."""

    timeout: float | None = Field(default=None, alias="request_timeout")
    """The maximum number of seconds to wait for a response."""

    client_options: dict | None = Field(
        default=None,
    )
    """A dictionary of client options to pass to the Google API client.

    Example: `api_endpoint`

    !!! warning

        If both `client_options['api_endpoint']` and `base_url` are specified,
        the `api_endpoint` in `client_options` takes precedence.
    """

    base_url: str | None = Field(
        default=None,
    )
    """Base URL to use for the API client.

    This is a convenience alias for `client_options['api_endpoint']`.

    !!! warning

        If `client_options` already contains an `api_endpoint`, this parameter will be
        ignored in favor of the existing value.
    """

    transport: str | None = Field(
        default=None,
        alias="api_transport",
    )
    """A string, one of: `['rest', 'grpc', 'grpc_asyncio']`.

    The Google client library defaults to `'grpc'` for sync clients.

    For async clients, `'rest'` is converted to `'grpc_asyncio'` unless
    a custom endpoint is specified.
    """

    additional_headers: dict[str, str] | None = Field(
        default=None,
    )
    """Key-value dictionary representing additional headers for the model call"""

    response_modalities: list[Modality] | None = Field(
        default=None,
    )
    """A list of modalities of the response"""

    thinking_budget: int | None = Field(
        default=None,
    )
    """Indicates the thinking budget in tokens."""

    media_resolution: MediaResolution | None = Field(
        default=None,
    )
    """Media resolution for the input media."""

    include_thoughts: bool | None = Field(
        default=None,
    )
    """Indicates whether to include thoughts in the response."""

    safety_settings: dict[HarmCategory, HarmBlockThreshold] | None = None
    """Default safety settings to use for all generations.

        !!! example

            ```python
            from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            ```
    """  # noqa: E501

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }


def get_user_agent(module: str | None = None) -> tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module: The module for a custom user agent header.
    """
    client_library_version = (
        f"{LC_GOOGLE_GENAI_VERSION}-{module}" if module else LC_GOOGLE_GENAI_VERSION
    )
    if os.environ.get(_TELEMETRY_ENV_VARIABLE_NAME):
        client_library_version += f"+{_TELEMETRY_TAG}"
    return client_library_version, f"langchain-google-genai/{client_library_version}"


def get_client_info(module: str | None = None) -> "ClientInfo":
    r"""Returns a client info object with a custom user agent header.

    Args:
        module: The module for a custom user agent header.
    """
    client_library_version, user_agent = get_user_agent(module)
    # TODO: remove ignore once google-auth has types.
    return ClientInfo(  # type: ignore[no-untyped-call]
        client_library_version=client_library_version,
        user_agent=user_agent,
    )


class SafetySettingDict(TypedDict):
    category: HarmCategory

    threshold: HarmBlockThreshold
