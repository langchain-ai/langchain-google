import copy
from collections.abc import AsyncIterator, Callable
from enum import Enum, auto
from typing import (
    Any,
    AsyncContextManager,
)

import httpx
from google import auth
from google.auth.credentials import Credentials
from google.auth.transport import requests as auth_requests
from httpx_sse import (
    EventSource,
    aconnect_sse,
    connect_sse,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_google_vertexai._base import _VertexAIBase

_MISTRAL_MODELS: list[str] = ["mistral-medium-3", "mistral-small-2503", "codestral-2"]
_LLAMA_MODELS: list[str] = [
    "meta/llama-3.2-90b-vision-instruct-maas",
    "meta/llama-3.3-70b-instruct-maas",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "meta/llama-4-scout-17b-16e-instruct-maas",
]


def _get_token(credentials: Credentials | None = None) -> str:
    """Returns a valid token for GCP auth."""
    credentials = (
        credentials
        if credentials
        else auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])[0]
    )
    request = auth_requests.Request()
    credentials.refresh(request)
    if not credentials.token:
        msg = "Couldn't retrieve a token!"
        raise ValueError(msg)
    return credentials.token


def _raise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = response.read().decode("utf-8")
        msg = (
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}"
        )
        raise httpx.HTTPStatusError(
            msg,
            request=response.request,
            response=response,
        )


async def _araise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = (await response.aread()).decode("utf-8")
        msg = (
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}"
        )
        raise httpx.HTTPStatusError(
            msg,
            request=response.request,
            response=response,
        )


async def _aiter_sse(
    event_source_mgr: AsyncContextManager[EventSource],
) -> AsyncIterator[dict]:
    """Iterate over the server-sent events."""
    async with event_source_mgr as event_source:
        await _araise_on_error(event_source.response)
        async for event in event_source.aiter_sse():
            if event.data == "[DONE]":
                return
            yield event.json()


class VertexMaaSModelFamily(str, Enum):
    LLAMA = auto()
    # https://cloud.google.com/blog/products/ai-machine-learning/llama-3-1-on-vertex-ai
    MISTRAL = auto()
    # https://cloud.google.com/blog/products/ai-machine-learning/codestral-and-mistral-large-v2-on-vertex-ai

    @classmethod
    def _missing_(cls, value: Any) -> "VertexMaaSModelFamily":
        model_name = value.lower()
        if model_name in _LLAMA_MODELS:
            return VertexMaaSModelFamily.LLAMA
        if model_name in _MISTRAL_MODELS:
            return VertexMaaSModelFamily.MISTRAL
        msg = f"Model {model_name} is not supported yet!"
        raise ValueError(msg)


class _BaseVertexMaasModelGarden(_VertexAIBase):
    append_tools_to_system_message: bool = False
    "Whether to append tools to the system message or not."
    model_family: VertexMaaSModelFamily | None = None
    timeout: int = 120

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        token = _get_token(credentials=self.credentials)
        endpoint = self.get_url()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "x-goog-api-client": self._library_version,
            "user_agent": self._user_agent,
        }
        self.client = httpx.Client(
            base_url=endpoint,
            headers=headers,
            timeout=self.timeout,
        )
        self.async_client = httpx.AsyncClient(
            base_url=endpoint,
            headers=headers,
            timeout=self.timeout,
        )

    @model_validator(mode="after")
    def validate_environment_model_garden(self) -> Self:
        """Validate that the python package exists in environment."""
        family = VertexMaaSModelFamily(self.model_name)
        self.model_family = family
        if family == VertexMaaSModelFamily.MISTRAL:
            model = self.model_name.split("@")[0] if self.model_name else None
            self.full_model_name = self.model_name
            self.model_name = model
        return self

    def _enrich_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fix params to be compliant with Vertex AI."""
        copy_params = copy.deepcopy(params)
        _ = copy_params.pop("safe_prompt", None)
        copy_params["model"] = self.model_name
        return copy_params

    def _get_url_part(self, stream: bool = False) -> str:
        if self.model_family == VertexMaaSModelFamily.MISTRAL:
            if stream:
                return (
                    f"publishers/mistralai/models/{self.full_model_name}"
                    ":streamRawPredict"
                )
            return f"publishers/mistralai/models/{self.full_model_name}:rawPredict"
        return "endpoints/openapi/chat/completions"

    def get_url(self) -> str:
        if self.model_family == VertexMaaSModelFamily.LLAMA:
            version = "v1beta1"
        else:
            version = "v1"
        return (
            f"https://{self.location}-aiplatform.googleapis.com/{version}/projects/"
            f"{self.project}/locations/{self.location}"
        )


def _create_retry_decorator(
    llm: _BaseVertexMaasModelGarden,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle exceptions."""
    errors = [httpx.RequestError, httpx.StreamError]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


async def acompletion_with_retry(
    llm: _BaseVertexMaasModelGarden,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        if "stream" not in kwargs:
            kwargs["stream"] = False
        stream = kwargs["stream"]
        if stream:
            # Llama and Mistral expect different "Content-Type" for streaming
            headers = {"Accept": "text/event-stream"}
            if headers_content_type := kwargs.pop("headers_content_type", None):
                headers["Content-Type"] = headers_content_type

            event_source = aconnect_sse(
                llm.async_client,
                "POST",
                llm._get_url_part(stream=True),
                json=kwargs,
                headers=headers,
            )
            return _aiter_sse(event_source)
        response = await llm.async_client.post(url=llm._get_url_part(), json=kwargs)
        await _araise_on_error(response)
        return response.json()

    kwargs = llm._enrich_params(kwargs)
    return await _completion_with_retry(**kwargs)


def completion_with_retry(llm: _BaseVertexMaasModelGarden, **kwargs):
    if "stream" not in kwargs:
        kwargs["stream"] = False
    stream = kwargs["stream"]
    kwargs = llm._enrich_params(kwargs)

    if stream:
        # Llama and Mistral expect different "Content-Type" for streaming
        headers = {"Accept": "text/event-stream"}
        if headers_content_type := kwargs.pop("headers_content_type", None):
            headers["Content-Type"] = headers_content_type

        def iter_sse():
            with connect_sse(
                llm.client,
                "POST",
                llm._get_url_part(stream=True),
                json=kwargs,
                headers=headers,
            ) as event_source:
                _raise_on_error(event_source.response)
                for event in event_source.iter_sse():
                    if event.data == "[DONE]":
                        return
                    yield event.json()

        return iter_sse()
    response = llm.client.post(url=llm._get_url_part(), json=kwargs)
    _raise_on_error(response)
    return response.json()
