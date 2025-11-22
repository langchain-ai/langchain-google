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
        alias="api_key",
        default_factory=secret_from_env(
            ["GOOGLE_API_KEY", "GEMINI_API_KEY"], default=None
        ),
    )
    """Google AI API key.

    If not specified, will check the env vars `GOOGLE_API_KEY` and `GEMINI_API_KEY` with
    precedence given to `GOOGLE_API_KEY`.
    """

    credentials: Any = None
    """The default custom credentials to use when making API calls.

    If not provided, credentials will be ascertained from the `GOOGLE_API_KEY`
    or `GEMINI_API_KEY` env vars with precedence given to `GOOGLE_API_KEY`.
    """

    temperature: float = 0.7
    """Run inference with this temperature.

    Must be within `[0.0, 2.0]`.

    !!! warning "Gemini 3.0+ models"

        Setting `temperature < 1.0` for Gemini 3.0+ models can cause infinite loops,
        degraded reasoning performance, and failure on complex tasks.

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

    To constrain the number of thinking tokens to use when generating a response, see
    the `thinking_budget` parameter.
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

    - **REST transport** (`transport="rest"`): Accepts full URLs with paths

        - `https://api.example.com/v1/path`
        - `https://webhook.site/unique-path`

    - **gRPC transports** (`transport="grpc"` or `transport="grpc_asyncio"`): Only
        accepts `hostname:port` format

        - `api.example.com:443`
        - `custom.googleapis.com:443`
        - `https://api.example.com` (auto-formatted to `api.example.com:443`)
        - NOT `https://webhook.site/path` (paths are not supported in gRPC)
        - NOT `api.example.com/path` (paths are not supported in gRPC)

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

    media_resolution: MediaResolution | None = Field(
        default=None,
    )
    """Media resolution for the input media.

    May be defined at the individual part level, allowing for mixed-resolution requests
    (e.g., images and videos of different resolutions in the same request).

    May be `'low'`, `'medium'`, or `'high'`.

    Can be set either per-part or globally for all media inputs in the request. To set
    globally, set in the `generation_config`.

    !!! warning "Model compatibility"

        Setting per-part media resolution requests to Gemini 2.5 models is not
        supported.
    """

    thinking_budget: int | None = Field(
        default=None,
    )
    """Indicates the thinking budget in tokens.

    Used to disable thinking for supported models (when set to `0`) or to constrain
    the number of tokens used for thinking.

    Dynamic thinking (allowing the model to decide how many tokens to use) is
    enabled when set to `-1`.

    More information, including per-model limits, can be found in the
    [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking#set-budget).
    """

    include_thoughts: bool | None = Field(
        default=None,
    )
    """Indicates whether to include thoughts in the response.

    !!! note

        This parameter is only applicable for models that support thinking.

        This does not disable thinking; to disable thinking, set `thinking_budget` to
        `0`. for supported models. See the `thinking_budget` parameter for more details.
    """

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
        # Either could contain the API key
        return {
            "google_api_key": "GOOGLE_API_KEY",
            "gemini_api_key": "GEMINI_API_KEY",
        }

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
