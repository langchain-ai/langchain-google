import os
from importlib import metadata
from typing import Any

from google.api_core.gapic_v1.client_info import ClientInfo
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_google_genai._enums import (
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
    SafetySetting,
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
    """Base class for Google Generative AI LLMs.

    ## Backend Selection

    This class supports both the Gemini Developer API and Google Cloud's Vertex AI
    Platform as backends. The backend used is selected **automatically** based on your
    authentication method:

    | Condition | Backend | Authentication |
    |-----------|---------|----------------|
    | API key provided | **Gemini Developer API** | `google_api_key`/`api_key` param or `GOOGLE_API_KEY`/`GEMINI_API_KEY` env var |
    | No API key | **Vertex AI** | `credentials` param or [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) |

    !!! tip "Quick Start"

        **Gemini Developer API** (simplest):

        ```python
        # Either set GOOGLE_API_KEY env var or pass api_key directly
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="MY_API_KEY")
        ```

        **Vertex AI** (enterprise):

        ```python
        # Ensure ADC is configured: gcloud auth application-default login
        # Either set GOOGLE_CLOUD_PROJECT env var or pass project directly
        # Location defaults to us-central1 or can be set via GOOGLE_CLOUD_LOCATION
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            project="my-project",
            # location="us-central1",
        )
        ```

    ## Environment Variables

    | Variable | Purpose | Backend |
    |----------|---------|---------|
    | `GOOGLE_API_KEY` | API key (primary) | Gemini Developer API |
    | `GEMINI_API_KEY` | API key (fallback) | Gemini Developer API |
    | `GOOGLE_CLOUD_PROJECT` | GCP project ID | Vertex AI |
    | `GOOGLE_CLOUD_LOCATION` | GCP region (default: `us-central1`) | Vertex AI |
    | `HTTPS_PROXY` | HTTP/HTTPS proxy URL | Both |
    | `SSL_CERT_FILE` | Custom SSL certificate file | Both |

    `GOOGLE_API_KEY` is checked first for backwards compatibility. (`GEMINI_API_KEY` was
    introduced later to better reflect the API's branding.)

    ## Proxy Configuration

    Set these before initializing:

    ```bash
    export HTTPS_PROXY='http://username:password@proxy_uri:port'
    export SSL_CERT_FILE='path/to/cert.pem'  # Optional: custom SSL certificate
    ```

    For SOCKS5 proxies or advanced proxy configuration, use the `client_args` parameter:

    ```python
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        client_args={"proxy": "socks5://user:pass@host:port"},
    )
    ```
    """  # noqa: E501

    # --- Client params ---

    google_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["GOOGLE_API_KEY", "GEMINI_API_KEY"], default=None
        ),
    )
    """API key for the Gemini Developer API.

    If not specified, will check the env vars `GOOGLE_API_KEY` and `GEMINI_API_KEY` with
    precedence given to `GOOGLE_API_KEY`.

    !!! note "Vertex AI"

        To use Vertex AI instead, either provide explicit `credentials` or ensure
        [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials)
        are configured on your system.

        When no API key is found, the SDK automatically uses Vertex AI with ADC (unless)
        custom `credentials` are provided.
    """

    credentials: Any = None
    """Custom credentials for Vertex AI authentication.

    When provided, forces Vertex AI backend (regardless of API key presence in
    `google_api_key`/`api_key`).

    Accepts a `google.auth.credentials.Credentials` object.

    If omitted and no API key is found, the SDK uses
    [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials).

    !!! example "Service account credentials"

        ```python
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            "path/to/service-account.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            credentials=credentials,
            project="my-project-id",
        )
        ```
    """

    project: str | None = Field(default=None)
    """Google Cloud project ID (**Vertex AI only**).

    Required when using Vertex AI.

    Falls back to `GOOGLE_CLOUD_PROJECT` env var if not provided.
    """

    location: str | None = Field(default=None)
    """Google Cloud region (**Vertex AI only**).

    If not provided, falls back to the `GOOGLE_CLOUD_LOCATION` env var, then
    `'us-central1'`.
    """

    base_url: str | dict | None = Field(default=None, alias="client_options")
    """Custom base URL for the API client.

    If not provided, defaults depend on the API being used:

    - **Gemini Developer API** (`api_key`/`google_api_key`): `https://generativelanguage.googleapis.com/`
    - **Vertex AI** (`credentials`): `https://{location}-aiplatform.googleapis.com/`

    !!! note

        Typed to accept `dict` to support backwards compatibility for the (now removed)
        `client_options` param.

        If a `dict` is passed in, it will only extract the `'api_endpoint'` key.
    """

    additional_headers: dict[str, str] | None = Field(
        default=None,
    )
    """Additional HTTP headers to include in API requests.

    Passed as `headers` to `HttpOptions` when creating the client.

    !!! example

        ```python
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            additional_headers={
                "X-Custom-Header": "value",
            },
        )
        ```
    """

    client_args: dict[str, Any] | None = Field(default=None)
    """Additional arguments to pass to the underlying HTTP client.

    Applied to both sync and async clients.

    !!! example "SOCKS5 proxy"

        ```python
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            client_args={"proxy": "socks5://user:pass@host:port"},
        )
        ```
    """

    transport: str | None = Field(
        default=None,
        alias="api_transport",
    )
    """Transport protocol for API calls. One of: `'rest'`, `'grpc'`, `'grpc_asyncio'`.

    !!! warning "Legacy parameter"

        This parameter is only used by `GoogleGenerativeAIEmbeddings` (which uses the
        legacy client).

        `ChatGoogleGenerativeAI` uses the new `google-genai` SDK which uses `httpx` for
        requests and does not support this parameter.
    """

    # --- Model / invocation params ---

    model: str = Field(...)
    """Model name to use."""

    temperature: float = 0.7
    """Run inference with this temperature.

    Must be within `[0.0, 2.0]`.

    !!! note "Automatic override for Gemini 3.0+ models"

        If `temperature` is not explicitly set and the model is Gemini 3.0 or later,
        it will be automatically set to `1.0` instead of the default `0.7` per the
        Google GenAI API best practices, as it can cause infinite loops, degraded
        reasoning performance, and failure on complex tasks.

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
            from google.genai.types import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            ```
    """  # noqa: E501

    @model_validator(mode="after")
    def _resolve_project_from_credentials(self) -> Self:
        """Extract project from credentials if not explicitly set.

        For backward compatibility with `langchain-google-vertexai`, which extracts
        `project_id` from credentials when not explicitly provided.
        """
        if self.project is None:
            if self.credentials and hasattr(self.credentials, "project_id"):
                self.project = self.credentials.project_id
        return self

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


SafetySettingDict = SafetySetting
