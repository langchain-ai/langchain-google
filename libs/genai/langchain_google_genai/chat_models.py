from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import inspect
import io
import json
import logging
import mimetypes
import os
import re
import threading
import uuid
import warnings
import wave
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Container,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import ContextVar
from difflib import get_close_matches
from enum import Enum
from operator import itemgetter
from typing import (
    Any,
    Literal,
    NoReturn,
    Self,
    cast,
)
from urllib.parse import urlparse

import filetype  # type: ignore[import-untyped]
import httpx2 as _httpx_v2
from google.genai import errors as _genai_errors
from google.genai.client import Client
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Blob,
    Candidate,
    CodeExecutionResult,
    Content,
    ExecutableCode,
    FileData,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerationConfig,
    HttpOptions,
    HttpRetryOptions,
    ImageConfig,
    Part,
    PrebuiltVoiceConfig,
    SafetySetting,
    ServiceTier,
    SpeechConfig,
    ThinkingConfig,
    ToolCodeExecution,
    ToolConfig,
    VideoMetadata,
    VoiceConfig,
)
from google.genai.types import (
    Outcome as CodeExecutionResultOutcome,
)
from google.genai.types import Tool as GoogleTool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
    is_openai_data_block,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages import content as types
from langchain_core.messages.ai import UsageMetadata, add_usage, subtract_usage
from langchain_core.messages.tool import invalid_tool_call, tool_call, tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    parse_tool_calls,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import is_typeddict

from langchain_google_genai._common import (
    GoogleGenerativeAIError,
    SafetySettingDict,
    _BaseGoogleGenerativeAI,
    _default_api_version,
    get_user_agent,
)
from langchain_google_genai._compat import (
    _MEDIA_PROCESSING_TOOL_NAME,
    _REASONING_BLOCK_TYPES,
    _classify_model_provider,
    _convert_from_v1_to_generativelanguage_v1beta,
    _is_file_uri_supported,
)
from langchain_google_genai._function_utils import (
    _tool_choice_to_tool_config,
    _ToolChoiceType,
    _ToolDict,
    convert_to_genai_function_declarations,
    is_basemodel_subclass_safe,
    tool_to_dict,
)
from langchain_google_genai._image_utils import (
    ImageBytesLoader,
    image_bytes_to_b64_string,
)
from langchain_google_genai._version import __version__
from langchain_google_genai.data._profiles import _PROFILES

try:
    from google.genai._gaos.lib.compat_errors import (
        APIConnectionError as _InteractionsAPIConnectionError,
    )
    from google.genai._gaos.lib.compat_errors import (
        APIError as _InteractionsAPIError,
    )
    from google.genai._gaos.lib.compat_errors import (
        APIStatusError as _InteractionsAPIStatusError,
    )
    from google.genai._gaos.lib.compat_errors import (
        APITimeoutError as _InteractionsAPITimeoutError,
    )
    from google.genai._gaos.lib.compat_errors import (
        AuthenticationError as _InteractionsAuthenticationError,
    )
    from google.genai._gaos.lib.compat_errors import (
        BadRequestError as _InteractionsBadRequestError,
    )
    from google.genai._gaos.lib.compat_errors import (
        GeminiNextGenAPIClientError as _GeminiNextGenAPIClientError,
    )
    from google.genai._gaos.lib.compat_errors import (
        InternalServerError as _InteractionsInternalServerError,
    )
    from google.genai._gaos.lib.compat_errors import (
        NotFoundError as _InteractionsNotFoundError,
    )
    from google.genai._gaos.lib.compat_errors import (
        PermissionDeniedError as _InteractionsPermissionDeniedError,
    )
    from google.genai._gaos.lib.compat_errors import (
        RateLimitError as _InteractionsRateLimitError,
    )
    from google.genai._gaos.lib.compat_errors import (
        UnprocessableEntityError as _InteractionsUnprocessableEntityError,
    )
    from google.genai._gaos.utils import RetryConfig as _InteractionsRetryConfig

    _HAS_GAOS_SDK = True
except ImportError:  # pragma: no cover - defensive fallback if private _gaos moves

    class _UnmatchedGaosError(Exception):
        """Placeholder exception when `google.genai._gaos` is unavailable."""

    _InteractionsAPIConnectionError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsAPIError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsAPIStatusError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsAPITimeoutError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsAuthenticationError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsBadRequestError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _GeminiNextGenAPIClientError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsInternalServerError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsNotFoundError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsPermissionDeniedError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsRateLimitError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsUnprocessableEntityError = _UnmatchedGaosError  # type: ignore[assignment,misc]
    _InteractionsRetryConfig = None  # type: ignore[assignment,misc]
    _HAS_GAOS_SDK = False

_ContinuationInterruptedError: type[BaseException] | None = getattr(
    _genai_errors, "ContinuationInterruptedError", None
)

try:
    import httpx as _httpx_v1
except ImportError:  # pragma: no cover - `httpx` is optional in google-genai>=3.0
    _httpx_v1 = None  # type: ignore[assignment]


def _collect_httpx_types(attr_name: str) -> tuple[type[Any], ...]:
    """Collect matching class types from `httpx2` and optional `httpx`."""
    found: list[type[Any]] = []
    for mod in (_httpx_v2, _httpx_v1):
        cls = getattr(mod, attr_name, None) if mod is not None else None
        if isinstance(cls, type) and cls not in found:
            found.append(cls)
    return tuple(found)


_HTTPX_HTTP_ERRORS: tuple[type[BaseException], ...] = _collect_httpx_types("HTTPError")
_HTTPX_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = _collect_httpx_types(
    "TimeoutException"
)
_HTTPX_REQUEST_ERRORS: tuple[type[BaseException], ...] = _collect_httpx_types(
    "RequestError"
)
_HTTPX_UNSUPPORTED_PROTOCOL_ERRORS: tuple[type[BaseException], ...] = (
    _collect_httpx_types("UnsupportedProtocol")
)
_HTTPX_TIMEOUT_TYPES: tuple[type[Any], ...] = _collect_httpx_types("Timeout")
_HTTPX_SYNC_CLIENT_TYPES: tuple[type[Any], ...] = _collect_httpx_types("Client")
_HTTPX_ASYNC_CLIENT_TYPES: tuple[type[Any], ...] = _collect_httpx_types("AsyncClient")
_HTTPX_SYNC_TRANSPORT_TYPES: tuple[type[Any], ...] = _collect_httpx_types(
    "BaseTransport"
)
_HTTPX_ASYNC_TRANSPORT_TYPES: tuple[type[Any], ...] = _collect_httpx_types(
    "AsyncBaseTransport"
)


def _detect_httpx_pkg(obj: Any) -> str | None:
    """Return `'httpx2'` or `'httpx'` if `obj` or an MRO base belongs to one."""
    if obj is None:
        return None
    for cls in type(obj).__mro__:
        top_pkg = cls.__module__.split(".", 1)[0]
        if top_pkg == "httpx2":
            return "httpx2"
        if top_pkg == "httpx" and _httpx_v1 is not None:
            return "httpx"
    return None


def _select_httpx_module(
    client_kwargs: Mapping[str, Any] | None = None,
    *,
    fallback_client: Any = None,
) -> Any:
    """Select `httpx2` or `httpx` matching any supplied transport/timeout/client."""
    if client_kwargs:
        transport = client_kwargs.get("transport")
        if transport is not None:
            pkg = _detect_httpx_pkg(transport)
            if pkg == "httpx" and _httpx_v1 is not None:
                return _httpx_v1
            if pkg == "httpx2":
                return _httpx_v2
        for key, val in client_kwargs.items():
            if key == "transport" or val is None:
                continue
            pkg = _detect_httpx_pkg(val)
            if pkg == "httpx" and _httpx_v1 is not None:
                return _httpx_v1
            if pkg == "httpx2":
                return _httpx_v2
    if fallback_client is not None:
        pkg = _detect_httpx_pkg(fallback_client)
        if pkg == "httpx" and _httpx_v1 is not None:
            return _httpx_v1
        if pkg == "httpx2":
            return _httpx_v2
    return _httpx_v2


def _create_httpx_client(
    client_kwargs: Mapping[str, Any],
    *,
    is_async: bool,
    fallback_client: Any = None,
) -> Any:
    """Instantiate an `httpx2` or `httpx` sync/async client matching `client_kwargs`."""
    http_mod = _select_httpx_module(client_kwargs, fallback_client=fallback_client)
    kw = dict(client_kwargs)
    timeout_val = kw.get("timeout")
    if (
        timeout_val is not None
        and isinstance(timeout_val, _HTTPX_TIMEOUT_TYPES)
        and not isinstance(timeout_val, http_mod.Timeout)
    ):
        kw["timeout"] = http_mod.Timeout(
            connect=timeout_val.connect,
            read=timeout_val.read,
            write=timeout_val.write,
            pool=timeout_val.pool,
        )
    return http_mod.AsyncClient(**kw) if is_async else http_mod.Client(**kw)


logger = logging.getLogger(__name__)

_FunctionDeclarationType = FunctionDeclaration | dict[str, Any] | Callable[..., Any]

_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY = (
    "__gemini_function_call_thought_signatures__"
)
_GEMINI_NATIVE_NON_STANDARD_TYPES = frozenset(
    {
        "media",
        "thinking",
        "executable_code",
        "code_execution_result",
    }
)
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"})
"""Candidate finish reasons that core translates to a refusal content block."""

_FINISH_REASONS: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
    "RECITATION": "content_filter",
    "SPII": "content_filter",
    "LANGUAGE": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "IMAGE_RECITATION": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
    "MALFORMED_RESPONSE": "error",
}

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


class ChatGoogleGenerativeAIError(GoogleGenerativeAIError):
    """Wrapper exception class for errors associated with the `Google GenAI` API.

    Raised when there are specific issues related to the Google GenAI API usage in the
    `ChatGoogleGenerativeAI` class, such as unsupported message types or roles.
    """


class GoogleContextOverflowError(ClientError, ContextOverflowError):
    """ClientError raised when input exceeds Google's context limit."""


class _ClientCleanup:
    """Close a client when the last model sharing it is collected."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._interactions_lock = threading.Lock()
        self._private_sync_source: Any = None
        self._private_sync_client_id: int | None = None
        self._private_sync_interactions: Any = None
        self._private_async_source: Any = None
        self._private_async_client_id: int | None = None
        self._private_async_interactions: Any = None

    def __eq__(self, other: object) -> bool:
        """Keep the internal cleanup token out of model equality semantics."""
        return isinstance(other, _ClientCleanup)

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _close_owned_sync_interactions(interactions_obj: Any) -> None:
        """Close any `_gaos`-allocated fallback sync `httpx.Client`."""
        seen: set[int] = set()
        for holder in (
            interactions_obj,
            getattr(interactions_obj, "parent_ref", None),
        ):
            sdk_cfg = getattr(holder, "sdk_configuration", None)
            if sdk_cfg is None or getattr(sdk_cfg, "client_supplied", True):
                continue
            http_c = getattr(sdk_cfg, "client", None)
            if http_c is not None and id(http_c) not in seen:
                seen.add(id(http_c))
                close_fn = getattr(http_c, "close", None)
                if callable(close_fn):
                    close_fn()

    @staticmethod
    async def _aclose_owned_async_interactions(interactions_obj: Any) -> None:
        """Close any `_gaos`-allocated fallback async `httpx.AsyncClient`."""
        seen: set[int] = set()
        for holder in (
            interactions_obj,
            getattr(interactions_obj, "parent_ref", None),
        ):
            sdk_cfg = getattr(holder, "sdk_configuration", None)
            if sdk_cfg is None or getattr(sdk_cfg, "async_client_supplied", True):
                continue
            async_c = getattr(sdk_cfg, "async_client", None)
            if async_c is not None and id(async_c) not in seen:
                seen.add(id(async_c))
                aclose_fn = getattr(async_c, "aclose", None)
                if callable(aclose_fn):
                    await aclose_fn()

    def _close_all_sync_owned(self) -> None:
        for ia in (
            self._private_sync_interactions,
            getattr(self._client, "_interactions", None),
        ):
            if ia is not None:
                try:
                    self._close_owned_sync_interactions(ia)
                except Exception:
                    pass

    async def _aclose_all_async_transports(self) -> None:
        first_exc: BaseException | None = None
        for ia in (
            self._private_async_interactions,
            getattr(getattr(self._client, "_aio", None), "_interactions", None),
        ):
            if ia is not None:
                try:
                    await self._aclose_owned_async_interactions(ia)
                except BaseException as exc:
                    if first_exc is None:
                        first_exc = exc
        try:
            await self._client.aio.aclose()
        except BaseException as exc:
            if first_exc is None:
                first_exc = exc
        if first_exc is not None:
            raise first_exc

    def register_async_loop(self) -> None:
        """Record the event loop that owns any async transports created later."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._async_loop is None:
            self._async_loop = running_loop

    async def aclose(self) -> None:
        """Close all transports on the event loop that owns the async client."""
        if self._closed:
            return

        running_loop = asyncio.get_running_loop()
        if self._async_loop is not None and running_loop is not self._async_loop:
            msg = "The async client must be closed on the event loop where it was used."
            raise RuntimeError(msg)

        self._async_loop = running_loop
        try:
            try:
                self._close_all_sync_owned()
            finally:
                self._client.close()
        finally:
            try:
                await self._aclose_all_async_transports()
            finally:
                self._closed = True

    def __del__(self) -> None:
        """Close sync and async transports without leaking cleanup exceptions."""
        if self._closed:
            return

        try:
            self._close_all_sync_owned()
        except Exception:
            pass

        try:
            self._client.close()
        except Exception:
            pass

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        except Exception:
            return

        if running_loop is not self._async_loop:
            return

        try:
            task = running_loop.create_task(self._aclose_all_async_transports())
            task.add_done_callback(self._consume_task_exception)
        except Exception:
            pass


# Inherit ChatGoogleGenerativeAIError for backward compatibility
class GoogleAuthenticationError(ChatGoogleGenerativeAIError, ModelAuthenticationError):
    """Google authentication error classified as a LangChain model error."""


class GooglePermissionDeniedError(
    ChatGoogleGenerativeAIError, ModelPermissionDeniedError
):
    """Google permission error classified as a LangChain model error."""


class GoogleInvalidRequestError(ChatGoogleGenerativeAIError, ModelInvalidRequestError):
    """Google invalid-request error classified as a LangChain model error."""


class GoogleModelNotFoundError(ChatGoogleGenerativeAIError, ModelNotFoundError):
    """Google not-found error classified as a LangChain model error."""


class GoogleRateLimitError(ChatGoogleGenerativeAIError, ModelRateLimitError):
    """Google rate-limit error classified as a LangChain model error."""


class GoogleAPIError(ServerError, ModelAPIError):
    """Google server error classified as a LangChain model error."""


class GoogleInteractionsContextOverflowError(
    GoogleContextOverflowError, ChatGoogleGenerativeAIError
):
    """Context-overflow error raised by the Gemini Interactions API."""


class GoogleInteractionsAPIError(GoogleAPIError, ChatGoogleGenerativeAIError):
    """Server or API error raised by the Gemini Interactions API."""


class GoogleConnectionError(
    ChatGoogleGenerativeAIError, ModelConnectionError, ModelAPIError
):
    """Connection error classified as a retryable LangChain model API error."""


class GoogleTimeoutError(ChatGoogleGenerativeAIError, ModelTimeoutError, ModelAPIError):
    """Timeout error classified as a retryable LangChain model API error."""


_NON_RETRYABLE_INTERACTIONS_STATUSES = frozenset(
    {
        "CANCELLED",
        "CANCELED",
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "RECITATION",
        "SPII",
        "IMAGE_SAFETY",
        "AUDIO_SAFETY",
        "VIDEO_SAFETY",
        "JAILBREAK",
        "CONTENT_FILTER",
        "FAILED_PRECONDITION",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
    }
)


def _is_retryable_interactions_api_error(exc: object) -> bool:
    """Return True when `exc` is a transient 5xx `GoogleInteractionsAPIError`."""
    if not isinstance(exc, GoogleInteractionsAPIError):
        return False
    status = str(getattr(exc, "status", "") or "").upper()
    if status in _NON_RETRYABLE_INTERACTIONS_STATUSES:
        return False
    code = getattr(exc, "code", None)
    if code is None:
        return True
    return 500 <= int(code) < 600 and int(code) != 501


class _RetryableServerErrorMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return _is_retryable_interactions_api_error(instance)


class _RetryableInteractionsAPIError(
    BaseException, metaclass=_RetryableServerErrorMeta
):
    """Dynamic `isinstance` predicate matching retryable 5xx Interactions API errors."""


_INTERACTIONS_RETRYABLE_ERRORS: list[type[BaseException]] = [
    _InteractionsRateLimitError,
    _InteractionsInternalServerError,
    _InteractionsAPIConnectionError,
    _InteractionsAPITimeoutError,
    GoogleTimeoutError,
    GoogleRateLimitError,
    GoogleConnectionError,
    _RetryableInteractionsAPIError,
]
"""Exception types raised by the Interactions SDK that are safe to retry."""

_INTERACTIONS_BASE_CATCHABLE_ERRORS: tuple[type[BaseException], ...] = (
    _GeminiNextGenAPIClientError,
    ClientError,
    ServerError,
    TimeoutError,
    *_HTTPX_HTTP_ERRORS,
)

_INTERACTIONS_CATCHABLE_ERRORS: tuple[type[BaseException], ...] = (
    (*_INTERACTIONS_BASE_CATCHABLE_ERRORS, _ContinuationInterruptedError)
    if _ContinuationInterruptedError is not None
    else _INTERACTIONS_BASE_CATCHABLE_ERRORS
)
"""SDK exception classes intercepted and classified by Interactions helpers."""

_INTERACTIONS_ONLY_TOOL_TYPES = frozenset({"file_search", "mcp_server", "retrieval"})
"""Interactions tool `type` tags that trigger automatic routing."""

_INTERACTIONS_SERVER_TOOL_TYPES = frozenset(
    {
        "google_search",
        "google_maps",
        "code_execution",
        "url_context",
        "computer_use",
        "file_search",
        "mcp_server",
        "retrieval",
    }
)
"""Server-side built-in tool `type` tags accepted by the Interactions API."""

_INTERACTIONS_AUTO_MODEL_PREFIXES = ("deep-research-", "antigravity-")
"""Normalized model/agent prefixes that require the Interactions API."""

_CONTEXT_OVERFLOW_PHRASES = (
    "exceeds the maximum number of tokens allowed",
    "token limit",
    "maximum context length",
    "context window",
    "too many tokens",
    "input token count",
)


_CLIENT_ERROR_TYPES: dict[int, type[ChatGoogleGenerativeAIError]] = {
    400: GoogleInvalidRequestError,
    401: GoogleAuthenticationError,
    403: GooglePermissionDeniedError,
    404: GoogleModelNotFoundError,
    408: GoogleTimeoutError,
    429: GoogleRateLimitError,
}
"""HTTP status codes the Google GenAI API reports, keyed to LangChain error types."""


def _is_context_overflow_text(text: str) -> bool:
    """Return whether an error string signals an input token/context limit overflow."""
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CONTEXT_OVERFLOW_PHRASES)


def _interactions_error_response_json(
    exc: BaseException, *, code: int, default_status: str
) -> dict[str, Any]:
    """Extract or synthesize a response JSON dict suitable for `ClientError`."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body
    return {
        "error": {
            "code": code,
            "message": str(exc),
            "status": default_status,
        }
    }


def _handle_interactions_error(
    exc: BaseException, model_name: str = "unknown"
) -> NoReturn:
    """Classify and raise a LangChain model error from an Interactions API exception.

    Maps `google.genai._gaos.lib.compat_errors` exceptions (and plain `ClientError` /
    `ServerError`) onto the `ChatGoogleGenerativeAIError` + `ModelError` hierarchy so
    upstream retry and context-compaction middleware behave identically across both
    Google transport backends.

    Args:
        exc: Exception raised by the Interactions SDK or stream transport.
        model_name: Name of the target model or agent for diagnostic context.

    Raises:
        GoogleInteractionsContextOverflowError: On HTTP 400 token/context limit errors.
        GoogleInvalidRequestError: On HTTP 400 or 422 invalid-request errors.
        GoogleAuthenticationError: On HTTP 401 authentication failures.
        GooglePermissionDeniedError: On HTTP 403 authorization failures.
        GoogleModelNotFoundError: On HTTP 404 unknown model/interaction errors.
        GoogleRateLimitError: On HTTP 429 rate-limit errors.
        GoogleTimeoutError: On client/request timeout errors.
        GoogleConnectionError: On transport connection errors.
        GoogleInteractionsAPIError: On HTTP 5xx or generic server/API errors.
        ChatGoogleGenerativeAIError: On any other unclassified status errors.
    """
    if isinstance(
        exc, (ChatGoogleGenerativeAIError, GoogleContextOverflowError, GoogleAPIError)
    ):
        raise exc

    if _ContinuationInterruptedError is not None and isinstance(
        exc, _ContinuationInterruptedError
    ):
        cause = exc.__cause__
        if isinstance(
            cause,
            (
                TimeoutError,
                *_HTTPX_TIMEOUT_ERRORS,
                _InteractionsAPITimeoutError,
            ),
        ):
            msg = f"Timeout calling model '{model_name}': {exc}"
            raise GoogleTimeoutError(msg) from exc
        if isinstance(cause, (*_HTTPX_HTTP_ERRORS, _InteractionsAPIConnectionError)):
            msg = f"Connection error calling model '{model_name}': {exc}"
            raise GoogleConnectionError(msg) from exc
        if cause is not None:
            try:
                _handle_interactions_error(cause, model_name=model_name)
            except (
                ChatGoogleGenerativeAIError,
                GoogleContextOverflowError,
                GoogleAPIError,
            ) as classified_cause:
                raise classified_cause from exc
        raise GoogleInteractionsAPIError(
            code=500,
            response_json=_interactions_error_response_json(
                exc, code=500, default_status="INTERNAL"
            ),
            response=None,
        ) from exc

    if isinstance(
        exc,
        (
            _InteractionsAPITimeoutError,
            TimeoutError,
            *_HTTPX_TIMEOUT_ERRORS,
        ),
    ):
        msg = f"Timeout calling model '{model_name}': {exc}"
        raise GoogleTimeoutError(msg) from exc

    if isinstance(exc, (_InteractionsAPIConnectionError, *_HTTPX_HTTP_ERRORS)):
        msg = f"Connection error calling model '{model_name}': {exc}"
        raise GoogleConnectionError(msg) from exc

    if isinstance(exc, ClientError):
        status_code_client = exc.code or 400
        if status_code_client == 400 and _is_context_overflow_text(str(exc)):
            raise GoogleInteractionsContextOverflowError(
                code=400,
                response_json=_interactions_error_response_json(
                    exc, code=400, default_status=exc.status or "INVALID_ARGUMENT"
                ),
                response=exc.response,
            ) from exc
        if status_code_client == 408:
            msg = f"Timeout calling model '{model_name}' (408): {exc}"
            raise GoogleTimeoutError(msg) from exc
        msg = f"Error calling model '{model_name}' ({exc.status}): {exc}"
        if error_type := _CLIENT_ERROR_TYPES.get(status_code_client):
            raise error_type(msg) from exc
        if status_code_client == 422:
            raise GoogleInvalidRequestError(msg) from exc
        raise ChatGoogleGenerativeAIError(msg) from exc

    if isinstance(exc, ServerError):
        code_srv = exc.code or 500
        raise GoogleInteractionsAPIError(
            code=code_srv,
            response_json=exc.details
            if isinstance(exc.details, dict)
            else _interactions_error_response_json(
                exc, code=code_srv, default_status=exc.status or "INTERNAL"
            ),
            response=exc.response,
        ) from exc

    status_code: int | None = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    combined_text = f"{exc} {body}" if body is not None else str(exc)
    response = getattr(exc, "response", None)

    if (
        isinstance(exc, _InteractionsBadRequestError) or status_code == 400
    ) and _is_context_overflow_text(combined_text):
        raise GoogleInteractionsContextOverflowError(
            code=400,
            response_json=_interactions_error_response_json(
                exc, code=400, default_status="INVALID_ARGUMENT"
            ),
            response=response,
        ) from exc

    if isinstance(
        exc, (_InteractionsBadRequestError, _InteractionsUnprocessableEntityError)
    ) or status_code in (400, 422):
        code_4xx = status_code if status_code is not None else 400
        msg = f"Error calling model '{model_name}' ({code_4xx}): {exc}"
        raise GoogleInvalidRequestError(msg) from exc

    if isinstance(exc, _InteractionsAuthenticationError) or status_code == 401:
        msg = f"Error calling model '{model_name}' (401): {exc}"
        raise GoogleAuthenticationError(msg) from exc

    if isinstance(exc, _InteractionsPermissionDeniedError) or status_code == 403:
        msg = f"Error calling model '{model_name}' (403): {exc}"
        raise GooglePermissionDeniedError(msg) from exc

    if isinstance(exc, _InteractionsNotFoundError) or status_code == 404:
        msg = f"Error calling model '{model_name}' (404): {exc}"
        raise GoogleModelNotFoundError(msg) from exc

    if status_code == 408:
        msg = f"Timeout calling model '{model_name}' (408): {exc}"
        raise GoogleTimeoutError(msg) from exc

    if isinstance(exc, _InteractionsRateLimitError) or status_code == 429:
        msg = f"Error calling model '{model_name}' (429): {exc}"
        raise GoogleRateLimitError(msg) from exc

    if isinstance(exc, _InteractionsInternalServerError) or (
        status_code is not None and 500 <= status_code < 600
    ):
        code_5xx = status_code if status_code is not None else 500
        raise GoogleInteractionsAPIError(
            code=code_5xx,
            response_json=_interactions_error_response_json(
                exc, code=code_5xx, default_status="INTERNAL"
            ),
            response=response,
        ) from exc

    if isinstance(exc, _InteractionsAPIStatusError):
        msg = f"Error calling model '{model_name}' ({status_code}): {exc}"
        raise ChatGoogleGenerativeAIError(msg) from exc

    if isinstance(exc, (_InteractionsAPIError, _GeminiNextGenAPIClientError)):
        code_api = status_code if status_code is not None else 500
        raise GoogleInteractionsAPIError(
            code=code_api,
            response_json=_interactions_error_response_json(
                exc, code=code_api, default_status="INTERNAL"
            ),
            response=response,
        ) from exc

    msg = f"Error calling model '{model_name}': {exc}"
    raise ChatGoogleGenerativeAIError(msg) from exc


def _clone_sdk_config(sdk_cfg: Any, **updates: Any) -> Any:
    """Shallow-copy `SDKConfiguration` while preserving `__dict__['_hooks']`."""
    cloned_cfg = copy.copy(sdk_cfg)
    for attr_k, attr_v in updates.items():
        setattr(cloned_cfg, attr_k, attr_v)
    return cloned_cfg


def _disable_sdk_internal_interactions_retries(interactions_client: Any) -> None:
    """Disable internal `_gaos` retries on a private Interactions client wrapper."""
    if _InteractionsRetryConfig is None or not type(
        interactions_client
    ).__module__.startswith("google.genai."):
        return
    sdk_cfg = getattr(interactions_client, "sdk_configuration", None)
    retry_cfg = getattr(sdk_cfg, "retry_config", None)
    if (
        sdk_cfg is not None
        and isinstance(retry_cfg, _InteractionsRetryConfig)
        and retry_cfg.strategy != "none"
    ):
        interactions_client.sdk_configuration = _clone_sdk_config(
            sdk_cfg, retry_config=_InteractionsRetryConfig("none", None, False)
        )


_FALLBACK_INTERACTIONS_LOCK = threading.Lock()
_FALLBACK_SYNC_INTERACTIONS_CACHE: weakref.WeakKeyDictionary[Any, Any] = (
    weakref.WeakKeyDictionary()
)
_FALLBACK_ASYNC_INTERACTIONS_CACHE: weakref.WeakKeyDictionary[Any, Any] = (
    weakref.WeakKeyDictionary()
)


def _merge_ephemeral_httpx_kwargs(
    base_client_args: Mapping[str, Any] | None,
    override_args: Mapping[str, Any],
    *,
    is_async: bool,
) -> dict[str, Any]:
    """Merge model-level `client_args` with per-request `http_options` overrides."""
    merged: dict[str, Any] = {}
    if isinstance(base_client_args, Mapping):
        for k, v in base_client_args.items():
            if k == "transport":
                ok_transport = (
                    isinstance(v, _HTTPX_ASYNC_TRANSPORT_TYPES)
                    if is_async
                    else isinstance(v, _HTTPX_SYNC_TRANSPORT_TYPES)
                )
                if v is not None and not ok_transport:
                    continue
            merged[k] = v
    merged.update(dict(override_args))
    merged.setdefault("follow_redirects", True)
    return merged


def _build_private_interactions_resource(target_client: Any, *, is_async: bool) -> Any:
    """Instantiate an isolated Interactions resource sharing `_api_client`'s pool."""
    api_client = getattr(target_client, "_api_client", None)
    if api_client is not None:
        try:
            if is_async:
                from google.genai._gaos.google_genai import (
                    AsyncGeminiNextGenInteractions,
                )

                priv_a = AsyncGeminiNextGenInteractions(api_client)
                _disable_sdk_internal_interactions_retries(priv_a)
                return priv_a
            from google.genai._gaos.google_genai import GeminiNextGenInteractions

            priv_s = GeminiNextGenInteractions(api_client)
            _disable_sdk_internal_interactions_retries(priv_s)
            return priv_s
        except ImportError:
            pass

    if not hasattr(target_client, "interactions"):
        msg = (
            "The installed 'google-genai' package lacks 'client.interactions'. "
            "Upgrade to 'google-genai>=3.2.0' to use the Gemini Interactions API."
        )
        raise ChatGoogleGenerativeAIError(msg)

    shared = target_client.interactions
    cloned = copy.copy(shared)
    _disable_sdk_internal_interactions_retries(cloned)
    return cloned


def _get_scoped_interactions_client(
    model_instance: Any,
    *,
    is_async: bool = False,
    base_url_override: str | None = None,
    client_args_override: Mapping[str, Any] | None = None,
    httpx_client_override: Any = None,
) -> tuple[Any, Any]:
    """Return `(interactions_client, ephemeral_httpx_client)` without mutating `client`.

    Preserves injected unit-test mocks (`MagicMock` / custom fakes on `_interactions`),
    keeps a thread-safe private Interactions wrapper per `ChatGoogleGenerativeAI` keyed
    by the active `target_client` so `model.client` reassignments invalidate stale
    wrappers cleanly, and creates a per-request scoped clone when `http_options`
    overrides `base_url`, `client_args`/`async_client_args`, or
    `httpx_client`/`httpx_async_client`.
    """
    target_client = model_instance.async_client if is_async else model_instance.client
    if target_client is None:
        msg = "Client not initialized."
        raise ValueError(msg)

    injected = getattr(target_client, "_interactions", None)
    cls_interactions = getattr(type(target_client), "interactions", None)
    has_patched_attr = "interactions" in getattr(target_client, "__dict__", {}) or (
        cls_interactions is not None and not isinstance(cls_interactions, property)
    )
    is_mock_client = (
        has_patched_attr
        or not type(target_client).__module__.startswith("google.genai.")
        or (
            injected is not None
            and not type(injected).__module__.startswith("google.genai.")
        )
    )
    if is_mock_client:
        return (
            target_client.interactions
            if has_patched_attr or injected is None
            else injected
        ), None

    cleanup = getattr(model_instance, "_client_cleanup", None)
    if not isinstance(cleanup, _ClientCleanup):
        priv_attrs = getattr(type(model_instance), "__private_attributes__", None)
        root_client = getattr(model_instance, "client", None)
        if (
            isinstance(priv_attrs, Mapping)
            and "_client_cleanup" in priv_attrs
            and isinstance(root_client, Client)
        ):
            with _FALLBACK_INTERACTIONS_LOCK:
                cleanup = getattr(model_instance, "_client_cleanup", None)
                if not isinstance(cleanup, _ClientCleanup):
                    cleanup = _ClientCleanup(root_client)
                    object.__setattr__(model_instance, "_client_cleanup", cleanup)

    target_id = id(target_client)
    base_ia: Any = None
    if isinstance(cleanup, _ClientCleanup):
        if is_async:
            cleanup.register_async_loop()
            src_attr = "_private_async_source"
            id_attr = "_private_async_client_id"
            ia_attr = "_private_async_interactions"
        else:
            src_attr = "_private_sync_source"
            id_attr = "_private_sync_client_id"
            ia_attr = "_private_sync_interactions"

        cached_ia = getattr(cleanup, ia_attr, None)
        if (
            cached_ia is not None
            and getattr(cleanup, src_attr, None) is target_client
            and getattr(cleanup, id_attr, None) == target_id
        ):
            base_ia = cached_ia
        else:
            stale_sync_ia: Any = None
            with cleanup._interactions_lock:
                cached_ia = getattr(cleanup, ia_attr, None)
                if (
                    cached_ia is not None
                    and getattr(cleanup, src_attr, None) is target_client
                    and getattr(cleanup, id_attr, None) == target_id
                ):
                    base_ia = cached_ia
                else:
                    if not is_async and cached_ia is not None:
                        stale_sync_ia = cached_ia
                    base_ia = _build_private_interactions_resource(
                        target_client, is_async=is_async
                    )
                    setattr(cleanup, src_attr, target_client)
                    setattr(cleanup, id_attr, target_id)
                    setattr(cleanup, ia_attr, base_ia)
                    if not is_async and isinstance(target_client, Client):
                        cleanup._client = target_client
            if stale_sync_ia is not None:
                try:
                    _ClientCleanup._close_owned_sync_interactions(stale_sync_ia)
                except Exception:
                    pass
    else:
        fb_cache = (
            _FALLBACK_ASYNC_INTERACTIONS_CACHE
            if is_async
            else _FALLBACK_SYNC_INTERACTIONS_CACHE
        )
        with _FALLBACK_INTERACTIONS_LOCK:
            try:
                base_ia = fb_cache.get(target_client)
            except TypeError:
                base_ia = None
            if base_ia is None:
                base_ia = _build_private_interactions_resource(
                    target_client, is_async=is_async
                )
                try:
                    fb_cache[target_client] = base_ia
                except TypeError:
                    pass

    if (
        not base_url_override
        and not client_args_override
        and httpx_client_override is None
    ):
        return base_ia, None

    sdk_cfg = getattr(base_ia, "sdk_configuration", None)
    if sdk_cfg is None or not hasattr(sdk_cfg, "server_url"):
        msg = (
            "Cannot apply per-request http_options base_url/client_args override: "
            "unsupported google-genai Interactions client configuration."
        )
        raise GoogleInvalidRequestError(msg)

    replacements: dict[str, Any] = {}
    ephemeral_http: Any = None
    if base_url_override:
        replacements["server_url"] = str(base_url_override).rstrip("/")
    if httpx_client_override is not None:
        if is_async:
            replacements["async_client"] = httpx_client_override
            replacements["async_client_supplied"] = True
        else:
            replacements["client"] = httpx_client_override
            replacements["client_supplied"] = True
    elif client_args_override:
        merged_args = _merge_ephemeral_httpx_kwargs(
            getattr(model_instance, "client_args", None),
            client_args_override,
            is_async=is_async,
        )
        fallback_http = (
            getattr(sdk_cfg, "async_client", None)
            if is_async
            else getattr(sdk_cfg, "client", None)
        )
        ephemeral_http = _create_httpx_client(
            merged_args,
            is_async=is_async,
            fallback_client=fallback_http,
        )
        if is_async:
            replacements["async_client"] = ephemeral_http
            replacements["async_client_supplied"] = True
        else:
            replacements["client"] = ephemeral_http
            replacements["client_supplied"] = True

    scoped = copy.copy(base_ia)
    scoped.sdk_configuration = _clone_sdk_config(sdk_cfg, **replacements)
    return scoped, ephemeral_http


def _create_interactions_retry_decorator(
    max_retries: int | None,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator covering retryable Interactions SDK exceptions."""
    attempts = max(1, max_retries) if max_retries is not None else 1
    return create_base_retry_decorator(
        error_types=_INTERACTIONS_RETRYABLE_ERRORS,
        max_retries=attempts,
        run_manager=run_manager,
    )


def _promote_retryable_status_error(exc: BaseException, model_name: str) -> None:
    """Promote retryable HTTP 408/429/5xx status errors into classified retry types."""
    if isinstance(
        exc, (ChatGoogleGenerativeAIError, GoogleContextOverflowError, GoogleAPIError)
    ):
        return
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 408:
        msg = f"Timeout calling model '{model_name}' (408): {exc}"
        raise GoogleTimeoutError(msg) from exc
    if code == 429:
        msg = f"Error calling model '{model_name}' (429): {exc}"
        raise GoogleRateLimitError(msg) from exc
    if isinstance(code, int) and 500 <= code < 600 and code != 501:
        details = getattr(exc, "details", None)
        raise GoogleInteractionsAPIError(
            code=code,
            response_json=details
            if isinstance(details, dict)
            else _interactions_error_response_json(
                exc,
                code=code,
                default_status=str(getattr(exc, "status", None) or "INTERNAL"),
            ),
            response=getattr(exc, "response", None),
        ) from exc


def _call_with_interactions_retry(
    func: Callable[..., Any],
    *,
    model_name: str,
    max_retries: int | None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **call_kwargs: Any,
) -> Any:
    """Invoke a sync Interactions SDK method with retry and error classification."""
    retry_decorator = _create_interactions_retry_decorator(
        max_retries=max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _invoke() -> Any:
        try:
            return func(**call_kwargs)
        except (_InteractionsAPIStatusError, ClientError, ServerError) as status_exc:
            _promote_retryable_status_error(status_exc, model_name)
            raise

    try:
        return _invoke()
    except _INTERACTIONS_CATCHABLE_ERRORS as exc:
        _handle_interactions_error(exc, model_name=model_name)


async def _acall_with_interactions_retry(
    func: Callable[..., Any],
    *,
    model_name: str,
    max_retries: int | None,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    **call_kwargs: Any,
) -> Any:
    """Invoke an async Interactions SDK method with retry and error classification."""
    retry_decorator = _create_interactions_retry_decorator(
        max_retries=max_retries, run_manager=run_manager
    )

    @retry_decorator
    async def _ainvoke() -> Any:
        try:
            return await func(**call_kwargs)
        except (_InteractionsAPIStatusError, ClientError, ServerError) as status_exc:
            _promote_retryable_status_error(status_exc, model_name)
            raise

    try:
        return await _ainvoke()
    except _INTERACTIONS_CATCHABLE_ERRORS as exc:
        _handle_interactions_error(exc, model_name=model_name)


def _redact_mcp_tools(tools: Any) -> Any:
    """Return a copy of `tools` with `headers` redacted on MCP server tool configs."""
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return tools
    changed = False
    redacted: list[Any] = []
    for tool_item in tools:
        candidate: Any = (
            tool_to_dict(tool_item)
            if isinstance(tool_item, GoogleTool) and tool_item.mcp_servers
            else tool_item
        )
        mcp_key = (
            "mcp_servers"
            if isinstance(candidate, Mapping)
            and isinstance(candidate.get("mcp_servers"), list)
            else (
                "mcpServers"
                if isinstance(candidate, Mapping)
                and isinstance(candidate.get("mcpServers"), list)
                else None
            )
        )
        if (
            isinstance(candidate, Mapping)
            and candidate.get("type") == "mcp_server"
            and candidate.get("headers") is not None
        ):
            copied = dict(candidate)
            copied["headers"] = "**REDACTED**"
            redacted.append(copied)
            changed = True
        elif isinstance(candidate, Mapping) and mcp_key is not None:
            sub_changed = False
            sub_list: list[Any] = []
            for srv in candidate[mcp_key]:
                if isinstance(srv, Mapping) and (
                    srv.get("headers") is not None
                    or (
                        isinstance(srv.get("streamable_http_transport"), Mapping)
                        and srv["streamable_http_transport"].get("headers") is not None
                    )
                ):
                    srv_copy = copy.deepcopy(dict(srv))
                    if srv_copy.get("headers") is not None:
                        srv_copy["headers"] = "**REDACTED**"
                    if (
                        isinstance(srv_copy.get("streamable_http_transport"), dict)
                        and srv_copy["streamable_http_transport"].get("headers")
                        is not None
                    ):
                        srv_copy["streamable_http_transport"]["headers"] = (
                            "**REDACTED**"
                        )
                    sub_list.append(srv_copy)
                    sub_changed = True
                else:
                    sub_list.append(srv)
            if sub_changed:
                outer = dict(candidate)
                outer[mcp_key] = sub_list
                redacted.append(outer)
                changed = True
            else:
                redacted.append(tool_item)
        else:
            redacted.append(tool_item)
    return redacted if changed else tools


def _handle_client_error(e: ClientError, request: dict[str, Any]) -> None:
    """Convert `ClientError` to a more specific exception when possible.

    Raises `GoogleContextOverflowError` (a `ContextOverflowError` subclass)
    when the error indicates that the input exceeded the model's token limit,
    so that upstream middleware (e.g. `SummarizationMiddleware`) can catch it
    and fall back to context compaction.

    Other recognized status codes are raised as the corresponding
    `langchain_core.exceptions.ModelError` subclass, which tells retry middleware
    whether another attempt could succeed. Those remain `ChatGoogleGenerativeAIError`
    subclasses, so existing handling continues to work.

    Args:
        e: The `ClientError` exception to handle.
        request: The request dict containing model info.

    Raises:
        GoogleContextOverflowError: When the error indicates a context overflow.
        ChatGoogleGenerativeAIError: For all other client errors.
    """
    error_str = str(e)
    if (
        "exceeds the maximum number of tokens allowed" in error_str
        or "token limit" in error_str.lower()
    ):
        raise GoogleContextOverflowError(
            code=e.code,
            response_json=e.details,
            response=e.response,
        ) from e
    model_name = request.get("model", "unknown")
    msg = f"Error calling model '{model_name}' ({e.status}): {e}"
    if error_type := _CLIENT_ERROR_TYPES.get(e.code):
        raise error_type(msg) from e
    raise ChatGoogleGenerativeAIError(msg) from e


def _handle_server_error(e: ServerError) -> None:
    """Re-raise a `ServerError` as its LangChain-classified equivalent.

    Args:
        e: The `ServerError` exception to handle.

    Raises:
        GoogleAPIError: Always, preserving the original type and message.
    """
    raise GoogleAPIError(
        code=e.code,
        response_json=e.details,
        response=e.response,
    ) from e


def _classified_stream(
    response: Iterator[GenerateContentResponse], request: dict[str, Any]
) -> Iterator[GenerateContentResponse]:
    """Yield stream chunks, classifying errors raised while the request runs.

    `generate_content_stream` is a generator, so the request is not issued until
    the returned iterator is advanced. Classifying at the call site would never
    fire; the errors surface here instead.

    Args:
        response: The chunk iterator returned by the SDK.
        request: The request dict containing model info.

    Yields:
        Each chunk the SDK produces.
    """
    try:
        yield from response
    except ClientError as e:
        _handle_client_error(e, request)
    except ServerError as e:
        _handle_server_error(e)


_ACTIVE_ASTREAM_SINK: ContextVar[list[Any] | None] = ContextVar(
    "_ACTIVE_ASTREAM_SINK", default=None
)
"""Per-invocation sink used by `astream` to await inner `aclose()` deterministically."""


async def _aclose_async_iterator(stream: Any) -> None:
    """Close `stream` via `contextlib.aclosing` when it provides an async `aclose`."""
    aclose_fn = getattr(stream, "aclose", None)
    if callable(aclose_fn):
        if inspect.iscoroutinefunction(aclose_fn):
            async with contextlib.aclosing(cast("AsyncGenerator[Any, None]", stream)):
                pass
        else:
            maybe_coro = aclose_fn()
            if inspect.isawaitable(maybe_coro):
                await maybe_coro


@contextlib.asynccontextmanager
async def _safe_aclosing(stream: Any) -> AsyncIterator[Any]:
    """Delegate to `contextlib.aclosing` while tolerating custom `AsyncIterator`s."""
    if callable(getattr(stream, "aclose", None)) and inspect.iscoroutinefunction(
        stream.aclose
    ):
        async with contextlib.aclosing(
            cast("AsyncGenerator[Any, None]", stream)
        ) as managed:
            yield managed
    else:
        try:
            yield stream
        finally:
            await _aclose_async_iterator(stream)


async def _aclassified_stream(
    response: AsyncIterator[GenerateContentResponse], request: dict[str, Any]
) -> AsyncIterator[GenerateContentResponse]:
    """Yield stream chunks, classifying errors raised while the request runs.

    The async counterpart to `_classified_stream`. Awaiting
    `generate_content_stream` only builds the async generator, so the request is
    issued once it is iterated.

    Args:
        response: The chunk iterator returned by the SDK.
        request: The request dict containing model info.

    Yields:
        Each chunk the SDK produces.
    """
    try:
        async with _safe_aclosing(response) as managed_response:
            async for chunk in managed_response:
                yield chunk
    except ClientError as e:
        _handle_client_error(e, request)
    except ServerError as e:
        _handle_server_error(e)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _base64_to_bytes(input_str: str) -> bytes:
    return base64.b64decode(input_str.encode("utf-8"))


# Gemini rejects empty parts, but a thought signature is itself a replayable payload.


def _decode_signature(sig: Any) -> bytes | None:
    """Decode a serialized thought signature to Gemini's byte representation.

    Checkpoints store signatures as base64 strings, while the SDK accepts bytes.
    """
    if isinstance(sig, str):
        return base64.b64decode(sig) or None
    if isinstance(sig, bytes):
        return sig or None
    return None


def _block_signature(block: Mapping[str, Any]) -> str | bytes | None:
    """Read a signature from its v0, v1, or Vertex location."""
    extras = block.get("extras")
    return (
        block.get("thought_signature")
        or block.get("signature")
        or (extras.get("signature") if isinstance(extras, Mapping) else None)
    )


def _is_redundant_v1beta_part(block: Any) -> bool:
    """Check whether a projected part duplicates a call or is truly empty.

    Signature-only parts are intentionally retained because Gemini requires their
    signatures to be replayed even when they contain no text.
    """
    if not isinstance(block, dict):
        return False
    if "function_call" in block:
        return True
    if block.get("text"):
        return False
    return set(block) <= {
        "text",
        "thought",
        "thought_signature",
    } and not block.get("thought_signature")


def _v1beta_dict_to_part(part: Mapping[str, Any]) -> Part | None:
    """Validate a type-less projected v1beta dictionary as a Gemini part."""
    if not set(part).intersection(Part.model_fields):
        return None
    return Part.model_validate(part)


def _merge_http_options(base: HttpOptions | None, override: HttpOptions) -> HttpOptions:
    """Merge a per-request `HttpOptions` over internally-derived options.

    `timeout` and `retry_options` are derived internally (from the model config
    or call-time `timeout`/`max_retries`); those are preserved unless the
    per-request `override` explicitly sets them.
    Any field explicitly set on `override` (e.g. `base_url`, `api_version`) wins,
    and `headers` from both are merged with `override` keys taking precedence.

    Args:
        base: Options derived from the model config (`timeout`/`retry_options`),
            or `None` when neither is set.
        override: Per-request options supplied at invocation time.

    Returns:
        The merged `HttpOptions`.
    """
    if base is None:
        return override
    merged = base.model_copy(deep=True)
    for field in override.model_fields_set:
        value = getattr(override, field)
        if field == "headers":
            merged.headers = {**(merged.headers or {}), **(value or {})}
        else:
            setattr(merged, field, value)
    return merged


# Vertex AI `generateContent` selects the service tier via this header rather than
# the request body field (`GenerateContentConfig.service_tier`).
_VERTEX_SERVICE_TIER_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"


# Starting with Gemini 3.6 Flash and Gemini 3.5 Flash-Lite, Google deprecated
# custom sampling parameters and disallows prefilling model turns. Per Google's
# docs these rules apply to "these models and all future Gemini model releases",
# but Gemini version numbers are not monotonic across variants (e.g. plain
# `gemini-3.5-flash` is *not* affected while `gemini-3.5-flash-lite` is), so the
# affected models are tracked by an explicit allowlist. Add future GA models here
# as they ship.
_FIXED_SAMPLING_AND_NO_PREFILL_MODELS = frozenset(
    {
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-4-carbon",
        "gemini-4-hydrogen",
        "gemini-4-neon",
    }
)
_CUSTOM_SAMPLING_PARAMETERS = ("temperature", "top_k", "top_p")


def _normalize_gemini_model_name(model_name: str | None) -> str:
    """Normalize model names by stripping resource paths and regional prefixes."""
    if not model_name:
        return ""
    normalized = model_name.lower().rsplit("/", 1)[-1]
    return re.sub(r"^[a-z]{2,}\.(?=(?:gemini|gemma)-)", "", normalized)


def _uses_fixed_sampling_and_disallows_prefill(model_name: str) -> bool:
    """Check whether a model uses fixed sampling and rejects model prefills."""
    normalized_model = _normalize_gemini_model_name(model_name)
    if not normalized_model:
        return False
    normalized_model = re.sub(r"-\d{3}$", "", normalized_model)
    return normalized_model in _FIXED_SAMPLING_AND_NO_PREFILL_MODELS


_GEMINI_MAJOR_VERSION = re.compile(r"gemini-(\d+)")


def _is_gemini_3_or_later(model_name: str) -> bool:
    """Checks if the model is Gemini 3 or later."""
    normalized = _normalize_gemini_model_name(model_name)
    if not normalized:
        return False
    return any(int(major) >= 3 for major in _GEMINI_MAJOR_VERSION.findall(normalized))


def _is_gemini_25_model(model_name: str) -> bool:
    """Checks if the model is a Gemini 2.5 model."""
    return "gemini-2.5" in _normalize_gemini_model_name(model_name)


_PROVIDER_RESOLVED_HOSTS = (
    "youtube.com",
    "youtu.be",
    "generativelanguage.googleapis.com",
)


def _provider_resolved_uri(block: Mapping[str, Any]) -> str | None:
    """Return a URI Gemini fetches itself, rather than one to download and inline."""
    file_id = block.get("file_id")
    if isinstance(file_id, str) and file_id:
        return file_id
    url = block.get("url")
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("gs://"):
        return url
    if url.startswith("files/") and not os.path.exists(url):
        return url
    host = urlparse(url).hostname or ""
    if any(host == h or host.endswith(f".{h}") for h in _PROVIDER_RESOLVED_HOSTS):
        return url
    return None


def _server_tool_name(tool_type: Any) -> str:
    """Derive a block type from a Gemini `ToolType` on a server tool part.

    Agentic video responses currently omit `tool_type`, so media processing is
    assumed when it is absent.

    Args:
        tool_type: `ToolType` enum member, raw string, or `None`.

    Returns:
        Lower-cased tool name.
    """
    if tool_type is None:
        return "media_processing"
    return str(getattr(tool_type, "value", tool_type)).lower()


def _validate_video_metadata(video_metadata: object) -> None:
    """Validate user-supplied video metadata before sending to the API.

    The Gemini API surfaces an opaque `500 Internal error` when video
    offsets are negative or `start_offset` exceeds `end_offset`. This
    helper checks the obvious cases up front and raises a clearer error
    so callers do not have to debug the underlying API response.

    Args:
        video_metadata: Raw `video_metadata` from a media part. Accepts
            a `Mapping` (e.g. `dict`) keyed by either `start_offset`/
            `end_offset` or their camelCase aliases, or a `VideoMetadata`
            Pydantic instance. Each offset may be a duration string like
            `"10s"`, a number of seconds, or a `{"seconds": int, "nanos":
            int}` mapping.

    Raises:
        ValueError: If an offset is negative, `start_offset` is greater
            than `end_offset`, or `video_metadata` is not a mapping or
            object exposing offset fields.
    """

    def _to_seconds(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if not value.endswith("s"):
                return None
            try:
                return float(value[:-1])
            except (TypeError, ValueError):
                return None
        if isinstance(value, Mapping):
            mapping = cast("Mapping[str, Any]", value)
            try:
                return (
                    float(mapping.get("seconds", 0))
                    + float(mapping.get("nanos", 0)) / 1e9
                )
            except (TypeError, ValueError):
                return None
        return None

    if isinstance(video_metadata, Mapping):
        mapping = cast("Mapping[str, Any]", video_metadata)
        raw_start = mapping.get("start_offset", mapping.get("startOffset"))
        raw_end = mapping.get("end_offset", mapping.get("endOffset"))
    elif hasattr(video_metadata, "start_offset") or hasattr(
        video_metadata, "end_offset"
    ):
        # Pydantic `VideoMetadata` instance — read fields via attribute
        # access so callers passing a constructed model don't regress.
        raw_start = getattr(video_metadata, "start_offset", None)
        raw_end = getattr(video_metadata, "end_offset", None)
    else:
        msg = (
            "video_metadata must be a mapping or a VideoMetadata-like object "
            f"with start_offset/end_offset, got {type(video_metadata).__name__}"
        )
        raise ValueError(msg)

    start = _to_seconds(raw_start)
    end = _to_seconds(raw_end)

    if start is not None and start < 0:
        msg = f"video_metadata.start_offset must be non-negative, got {start}s"
        raise ValueError(msg)
    if end is not None and end < 0:
        msg = f"video_metadata.end_offset must be non-negative, got {end}s"
        raise ValueError(msg)
    if start is not None and end is not None and start > end:
        msg = (
            f"video_metadata.start_offset ({start}s) must not exceed "
            f"video_metadata.end_offset ({end}s)"
        )
        raise ValueError(msg)


def _convert_to_parts(
    raw_content: str | Sequence[str | dict],
    model: str | None = None,
    *,
    allow_v1beta_dicts: bool = False,
) -> list[Part]:
    """Converts LangChain message content into `generativelanguage_v1beta` parts.

    Used when preparing Human, System and AI messages for sending to the API.

    Handles both legacy (pre-v1) dict-based content blocks and v1 `ContentBlock`
    objects.

    Args:
        raw_content: Message content to convert.
        model: Model name used for version-specific conversion behavior.
        allow_v1beta_dicts: Whether to validate type-less projected v1beta mappings.

    Returns:
        Gemini parts representing the message content.
    """
    content = [raw_content] if isinstance(raw_content, str) else raw_content
    image_loader = ImageBytesLoader()

    parts = []
    # Iterate over each item in the content list, constructing a list of Parts
    for part in content:
        if isinstance(part, str):
            parts.append(Part(text=part))
        elif isinstance(part, Mapping):
            if "type" in part:
                if part["type"] == "text":
                    # Either old dict-style CC text block or new TextContentBlock
                    parts.append(
                        Part(
                            text=part["text"],
                            thought_signature=_decode_signature(_block_signature(part)),
                        )
                    )
                elif part.get("type") == "file" and "file_id" in part:
                    # Handle FileContentBlock with file_id (uploaded file reference)
                    mime_type = part.get("mime_type", "application/octet-stream")
                    file_part_kwargs: dict[str, Any] = {
                        "file_data": FileData(
                            file_uri=part["file_id"], mime_type=mime_type
                        )
                    }
                    if "media_processing" in part:
                        file_part_kwargs["media_processing"] = part["media_processing"]
                    parts.append(Part(**file_part_kwargs))
                elif is_data_content_block(part):
                    # Handle both legacy LC blocks (with `source_type`) and blocks >= v1

                    resolved_uri = (
                        None if "source_type" in part else _provider_resolved_uri(part)
                    )
                    part_kwargs: dict[str, Any] = {}
                    if resolved_uri is not None:
                        part_kwargs = {
                            "file_data": FileData(
                                file_uri=resolved_uri, mime_type=part.get("mime_type")
                            )
                        }
                        bytes_ = b""
                    elif "source_type" in part:
                        # Catch legacy v0 formats
                        # Safe since v1 content blocks don't have `source_type` key
                        if part["source_type"] == "url":
                            bytes_ = image_loader._bytes_from_url(part["url"])
                        elif part["source_type"] == "base64":
                            bytes_ = base64.b64decode(part["data"])
                        else:
                            # Unable to support IDContentBlock
                            msg = "source_type must be url or base64."
                            raise ValueError(msg)
                    elif "url" in part:
                        # v1 multimodal block w/ URL
                        bytes_ = image_loader._bytes_from_url(part["url"])
                    elif "base64" in part:
                        # v1 multimodal block w/ base64
                        bytes_ = base64.b64decode(part["base64"])
                    else:
                        msg = (
                            "Data content block must contain 'url', 'base64', or "
                            "'data' field."
                        )
                        raise ValueError(msg)

                    if resolved_uri is None:
                        mime_type = part.get("mime_type")
                        if not mime_type:
                            # Guess MIME type based on data field if not provided
                            source = cast(
                                "str",
                                part.get("url")
                                or part.get("base64")
                                or part.get("data"),
                            )
                            mime_type, _ = mimetypes.guess_type(source)
                            if not mime_type:
                                # Last resort - try to guess based on file bytes
                                kind = filetype.guess(bytes_)
                                if kind:
                                    mime_type = kind.mime
                        blob_kwargs: dict[str, Any] = {
                            "data": bytes_,
                        }
                        if mime_type:
                            blob_kwargs["mime_type"] = mime_type

                        part_kwargs = {"inline_data": Blob(**blob_kwargs)}
                    if "media_resolution" in part:
                        if model and _is_gemini_25_model(model):
                            warnings.warn(
                                "Setting per-part media resolution requests to "
                                "Gemini 2.5 models and older is not supported. The "
                                "media_resolution parameter will be ignored.",
                                UserWarning,
                                stacklevel=2,
                            )
                        elif model and _is_gemini_3_or_later(model):
                            part_kwargs["media_resolution"] = {
                                "level": part["media_resolution"]
                            }
                    if "media_processing" in part:
                        part_kwargs["media_processing"] = part["media_processing"]
                    thought_signature = None
                    if "extras" in part and isinstance(part["extras"], dict):
                        sig = part["extras"].get("signature")
                        if isinstance(sig, str):
                            thought_signature = base64.b64decode(sig)
                        elif isinstance(sig, bytes):
                            thought_signature = sig

                        if thought_signature:
                            pass

                    parts.append(Part(**part_kwargs))
                elif part["type"] == "image_url":
                    # Chat Completions image format
                    img_url = part["image_url"]
                    if isinstance(img_url, dict):
                        if "url" not in img_url:
                            msg = f"Unrecognized message image format: {img_url}"
                            raise ValueError(msg)
                        img_url = img_url["url"]
                    # Check for thought_signature in extras
                    # (needed for multi-turn image editing/usage)
                    thought_sig = None
                    image_part = image_loader.load_part(img_url)
                    if "extras" in part and isinstance(part["extras"], dict):
                        sig = part["extras"].get("signature")
                        if isinstance(sig, str):
                            image_part.thought_signature = base64.b64decode(sig)
                        elif isinstance(sig, bytes):
                            image_part.thought_signature = sig
                    if thought_sig:
                        image_part.thought_signature = thought_sig
                    parts.append(image_part)
                elif part["type"] in ("media", "file_data"):
                    # Handle `media` following pattern established in LangChain.js
                    # https://github.com/langchain-ai/langchainjs/blob/e536593e2585f1dd7b0afc187de4d07cb40689ba/libs/langchain-google-common/src/utils/gemini.ts#L93-L106
                    # Inline bytes carry no type information, so a MIME type is
                    # required. A `file_uri` is resolved server-side and the API
                    # infers the type.
                    mime_type = part.get("mime_type")
                    media_part_kwargs: dict[str, Any] = {}

                    if "data" in part:
                        if not mime_type:
                            msg = f"Missing mime_type in media part: {part}"
                            raise ValueError(msg)
                        data = part["data"]
                        if isinstance(data, str):
                            clean_data = re.sub(r"\s+", "", data)
                            data_validation_msg = "Data should be valid base64"
                            if (
                                not re.match(r"^[A-Za-z0-9+/]*={0,2}$", clean_data)
                                or len(clean_data) % 4 != 0
                            ):
                                raise ValueError(data_validation_msg)
                            try:
                                data = base64.b64decode(clean_data)
                            except Exception:
                                raise ValueError(data_validation_msg)
                        media_part_kwargs["inline_data"] = Blob(
                            data=data, mime_type=mime_type
                        )
                    elif "file_uri" in part:
                        media_part_kwargs["file_data"] = FileData(
                            file_uri=part["file_uri"], mime_type=mime_type
                        )
                    else:
                        msg = f"Media part must have either data or file_uri: {part}"
                        raise ValueError(msg)
                    if "video_metadata" in part:
                        _validate_video_metadata(part["video_metadata"])
                        metadata = VideoMetadata.model_validate(part["video_metadata"])
                        media_part_kwargs["video_metadata"] = metadata
                        if "extras" in part and isinstance(part["extras"], dict):
                            sig = part["extras"].get("signature")
                            if sig and isinstance(sig, str):
                                media_part_kwargs["thought_signature"] = (
                                    base64.b64decode(sig)
                                )

                    if "media_resolution" in part:
                        if model and _is_gemini_25_model(model):
                            warnings.warn(
                                "Setting per-part media resolution requests to "
                                "Gemini 2.5 models and older is not supported. The "
                                "media_resolution parameter will be ignored.",
                                UserWarning,
                                stacklevel=2,
                            )
                        elif model and _is_gemini_3_or_later(model):
                            media_part_kwargs["media_resolution"] = {
                                "level": part["media_resolution"]
                            }
                    if "media_processing" in part:
                        media_part_kwargs["media_processing"] = part["media_processing"]
                    if "extras" in part and isinstance(part["extras"], dict):
                        sig = part["extras"].get("signature")
                        if isinstance(sig, str):
                            media_part_kwargs["thought_signature"] = base64.b64decode(
                                sig
                            )
                        elif isinstance(sig, bytes):
                            media_part_kwargs["thought_signature"] = sig

                    parts.append(Part(**media_part_kwargs))
                elif part["type"] == "thinking":
                    # Pre-existing thinking block format that we continue to store as
                    parts.append(
                        Part(
                            text=part["thinking"],
                            thought=True,
                            thought_signature=_decode_signature(_block_signature(part)),
                        )
                    )
                elif part["type"] == "reasoning":
                    # ReasoningContentBlock (when output_version = "v1")
                    thought_sig = _decode_signature(_block_signature(part))
                    reasoning_text = part.get("reasoning")
                    if reasoning_text is None:
                        # Foreign-provider reasoning block without a `reasoning`
                        # key (e.g. OpenAI's `summary` shape). Extract summary
                        # text when possible; otherwise drop the block.
                        summary = part.get("summary")
                        if isinstance(summary, list):
                            summary_texts = [
                                text.strip()
                                for item in summary
                                if isinstance(item, dict)
                                and isinstance((text := item.get("text")), str)
                                and text.strip()
                            ]
                            reasoning_text = " ".join(summary_texts)
                        if not reasoning_text:
                            continue
                    parts.append(
                        Part(
                            text=reasoning_text,
                            thought=True,
                            thought_signature=thought_sig,
                        )
                    )
                elif part["type"] == "server_tool_call":
                    call_extras = part.get("extras")
                    step_tag = (
                        call_extras.get("step_type")
                        if isinstance(call_extras, Mapping)
                        else None
                    )
                    if part.get("name") == _MEDIA_PROCESSING_TOOL_NAME:
                        # Media processing steps are returned by the server for
                        # observability but cannot be replayed: the GenerateContent API
                        # rejects echoed steps with "Tool type of tool_call part does
                        # not match with tool call context". Video context is preserved
                        # instead by keeping the original media part in history.
                        continue
                    if part.get("name") == "code_interpreter":
                        args = part.get("args", {})
                        code = args.get("code", "")
                        language = args.get("language", "python")
                        executable_code_part = Part(
                            executable_code=ExecutableCode(language=language, code=code)
                        )
                        parts.append(executable_code_part)
                    elif (
                        part.get("name")
                        in {
                            "google_search",
                            "google_maps",
                            "url_context",
                            "file_search",
                            "retrieval",
                            "remote_mcp",
                        }
                        or step_tag
                        in {
                            "google_search_call",
                            "google_maps_call",
                            "url_context_call",
                            "file_search_call",
                            "mcp_server_tool_call",
                            "retrieval_call",
                            "processing_call",
                        }
                        or (
                            isinstance(call_extras, Mapping)
                            and bool(call_extras.get("server_name"))
                        )
                    ):
                        # Interactions server tool calls other than code execution have
                        # no echoed Part equivalent on GenerateContent.
                        continue
                    else:
                        warnings.warn(
                            f"Server tool call with name '{part.get('name')}' is not "
                            "currently supported by Google GenAI. Only "
                            "'code_interpreter' is supported.",
                            stacklevel=2,
                        )
                elif part["type"] == "executable_code":
                    # Legacy executable_code format (backward compat)
                    if "executable_code" not in part or "language" not in part:
                        msg = (
                            "Executable code part must have 'code' and 'language' "
                            f"keys, got {part}"
                        )
                        raise ValueError(msg)
                    executable_code_part = Part(
                        executable_code=ExecutableCode(
                            language=part["language"], code=part["executable_code"]
                        )
                    )
                    parts.append(executable_code_part)
                elif part["type"] == "server_tool_result":
                    extras = part.get("extras")
                    if isinstance(extras, Mapping) and (
                        extras.get("block_type") not in (None, "code_execution_result")
                        or extras.get("step_type")
                        not in (None, "code_execution_result")
                    ):
                        # Non-code server tool results (media_processing, MCP, Google
                        # Search, Maps, URL context, file search, retrieval) have no
                        # echoed Part equivalent on GenerateContent and must never be
                        # fabricated as orphaned CodeExecutionResult parts.
                        continue
                    output = part.get("output", "")
                    status = part.get("status", "success")
                    outcome = (
                        CodeExecutionResultOutcome.OUTCOME_OK
                        if status == "success"
                        else CodeExecutionResultOutcome.OUTCOME_FAILED
                    )
                    # Check extras for original outcome if available
                    if isinstance(extras, Mapping) and "outcome" in extras:
                        outcome = extras["outcome"]
                    code_execution_result_part = Part(
                        code_execution_result=CodeExecutionResult(
                            output=str(output), outcome=outcome
                        )
                    )
                    parts.append(code_execution_result_part)
                elif part["type"] == "code_execution_result":
                    # Legacy code_execution_result format (backward compat)
                    if "code_execution_result" not in part:
                        msg = (
                            "Code execution result part must have "
                            f"'code_execution_result', got {part}"
                        )
                        raise ValueError(msg)
                    if "outcome" in part:
                        raw_outcome = part["outcome"]
                        # Convert integer outcome to enum if needed
                        # (for backward compat)
                        if isinstance(raw_outcome, int):
                            if raw_outcome == 1:
                                outcome = CodeExecutionResultOutcome.OUTCOME_OK
                            elif raw_outcome == 2:
                                outcome = CodeExecutionResultOutcome.OUTCOME_FAILED
                            else:
                                outcome = CodeExecutionResultOutcome.OUTCOME_UNSPECIFIED
                        else:
                            outcome = raw_outcome
                    else:
                        # Backward compatibility
                        outcome = CodeExecutionResultOutcome.OUTCOME_OK
                    code_execution_result_part = Part(
                        code_execution_result=CodeExecutionResult(
                            outcome=outcome,
                            output=part["code_execution_result"],
                        )
                    )
                    parts.append(code_execution_result_part)
                elif part["type"] == "non_standard":
                    value = part.get("value")
                    if (
                        isinstance(value, Mapping)
                        and value.get("type") in _GEMINI_NATIVE_NON_STANDARD_TYPES
                    ):
                        # Core wraps Gemini-native blocks it cannot standardize.
                        # Unwrap only known shapes; malformed native values must
                        # still raise rather than silently changing the request.
                        parts.extend(
                            _convert_to_parts(
                                [dict(value)],
                                model=model,
                                allow_v1beta_dicts=allow_v1beta_dicts,
                            )
                        )
                    else:
                        logger.warning(
                            "Dropping non-standard content block that cannot be "
                            "represented as a Gemini part (inner type: %s).",
                            value.get("type") if isinstance(value, Mapping) else None,
                        )
                elif part["type"] == "refusal":
                    # Gemini has no refusal input part.
                    continue
                else:
                    msg = f"Unrecognized message part type: {part['type']}."
                    raise ValueError(msg)
            elif (
                allow_v1beta_dicts
                and (v1beta_part := _v1beta_dict_to_part(part)) is not None
            ):
                parts.append(v1beta_part)
            elif part.keys() == {"cachePoint"}:
                pass
            else:
                # Yolo. The input message content doesn't have a `type` key
                logger.warning(
                    "Unrecognized message part format. Assuming it's a text part."
                )
                parts.append(Part(text=str(part)))
        else:
            msg = "Unknown error occurred while converting LC message content to parts."
            raise ChatGoogleGenerativeAIError(msg)
    return parts


def _contains_json_ref(value: Any) -> bool:
    """Return whether a parsed JSON value has a `$ref` key at any depth."""
    if isinstance(value, Mapping):
        return "$ref" in value or any(_contains_json_ref(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_json_ref(item) for item in value)
    return False


def _convert_tool_message_to_parts(
    message: ToolMessage | FunctionMessage,
    name: str | None = None,
    model: str | None = None,
) -> list[Part]:
    """Converts a tool or function message to a Google `Part`."""
    # Legacy agent stores tool name in message.additional_kwargs instead of message.name
    name = message.name or name or message.additional_kwargs.get("name")
    response: Any
    parts: list[Part] = []
    if isinstance(message.content, list):
        media_blocks = []
        other_blocks = []
        for block in message.content:
            if isinstance(block, dict) and (
                is_data_content_block(block) or is_openai_data_block(block)
            ):
                media_blocks.append(block)
            else:
                other_blocks.append(block)
        parts.extend(_convert_to_parts(media_blocks, model=model))
        response = other_blocks

    elif not isinstance(message.content, str):
        response = message.content
    else:
        try:
            response = json.loads(message.content)
        except json.JSONDecodeError:
            response = message.content  # leave as str representation
    if _contains_json_ref(response):
        # Gemini resolves `$ref` keys in a function response against the
        # display names of its multimodal parts and rejects the request when
        # none match, so tool output such as JSON schemas is sent as text.
        response = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(response, default=str)
        )
    part = Part(
        function_response=FunctionResponse(
            name=name,
            response=(
                {"output": response} if not isinstance(response, dict) else response
            ),
        )
    )
    parts.append(part)
    return parts


def _get_ai_message_tool_messages_parts(
    tool_messages: Sequence[ToolMessage],
    ai_message: AIMessage,
    model: str | None = None,
) -> list[Part]:
    """Conversion.

    Finds relevant tool messages for the AI message and converts them to a single list
    of `Part`s.
    """
    # We are interested only in the tool messages that are part of the AI message
    tool_calls_ids = {tool_call["id"]: tool_call for tool_call in ai_message.tool_calls}
    parts = []
    for _i, message in enumerate(tool_messages):
        if not tool_calls_ids:
            break
        if message.tool_call_id in tool_calls_ids:
            tool_call = tool_calls_ids[message.tool_call_id]
            message_parts = _convert_tool_message_to_parts(
                message, name=tool_call.get("name"), model=model
            )
            parts.extend(message_parts)
            # remove the id from the dict, so that we do not iterate over it again
            tool_calls_ids.pop(message.tool_call_id)
    return parts


# Placeholder injected by `_parse_chat_history` when a function-call part for a
# Gemini 3+ model lacks a thought signature (e.g. replayed history or a
# cross-provider fallback). Carrying the documented bypass sentinel makes the
# API skip thought-signature validation for that part; a real signature copied
# from an unrelated response is rejected as invalid.
#
# https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures
SKIP_THOUGHT_SIGNATURE_VALIDATOR = "skip_thought_signature_validator"

# Kept for backwards compatibility with imports of the previous constant.
DUMMY_THOUGHT_SIGNATURE = SKIP_THOUGHT_SIGNATURE_VALIDATOR.encode("ascii")


def _convert_to_parts_lenient(
    content: list[Any],
    model: str | None = None,
    *,
    allow_v1beta_dicts: bool = False,
) -> list[Part]:
    """Convert foreign content one block at a time, dropping invalid blocks.

    Per-block conversion preserves valid neighbors without retrying media downloads
    that occurred before a failing block.

    Args:
        content: Message content blocks to convert.
        model: Model name used for version-specific conversion behavior.
        allow_v1beta_dicts: Whether to validate type-less projected v1beta mappings.

    Returns:
        Valid Gemini parts from the supplied content.
    """
    parts: list[Part] = []
    for block in content:
        try:
            parts.extend(
                _convert_to_parts(
                    [block],
                    model=model,
                    allow_v1beta_dicts=allow_v1beta_dicts,
                )
            )
        except (ValueError, ChatGoogleGenerativeAIError) as exc:
            logger.warning(
                "Dropping content block that cannot be represented as a Gemini "
                "part (type: %s): %s",
                block.get("type") if isinstance(block, Mapping) else type(block),
                exc,
            )
    return parts


def _as_gemini_media_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """Re-tag raw Gemini file data for the shared media converter."""
    return {
        "type": "media",
        "file_uri": block["file_uri"],
        "mime_type": block.get("mime_type", "application/octet-stream"),
    }


def _is_rebuilt_tool_call_block(block: Mapping[str, Any]) -> bool:
    """Check whether a block is rebuilt from `message.tool_calls`."""
    return block.get("type") in (
        "tool_call",
        "tool_call_chunk",
        "invalid_tool_call",
        "function_call_signature",
    )


def _function_call_signatures_from_content(
    message: AIMessage,
) -> dict[int, str | bytes]:
    """Index legacy function-call signature sidecars by tool call."""
    if _classify_model_provider(
        message.response_metadata.get("model_provider")
    ) == "foreign" or not isinstance(message.content, list):
        return {}
    signatures: dict[int, str | bytes] = {}
    signature_ordinal = 0
    for block in message.content:
        if (
            not isinstance(block, Mapping)
            or block.get("type") != "function_call_signature"
        ):
            continue
        signature = block.get("signature")
        if not isinstance(signature, (str, bytes)) or not signature:
            continue
        # Early signature sidecars did not include an index and were appended after
        # other content. Their position in the full content list therefore does not
        # identify the corresponding entry in `message.tool_calls`.
        tool_call_index = block.get("index", signature_ordinal)
        signature_ordinal += 1
        if isinstance(tool_call_index, int):
            signatures[tool_call_index] = signature
    return signatures


def _is_unsupported_foreign_server_tool(
    block: Mapping[str, Any], code_interpreter_call_ids: Container[str]
) -> bool:
    """Check whether a foreign server-tool block has a Gemini equivalent.

    Code-interpreter calls map to executable code; their results are retained only
    when the corresponding call was retained.
    """
    block_type = block.get("type")
    if block_type == "server_tool_call":
        return block.get("name") != "code_interpreter"
    if block_type == "server_tool_result":
        return block.get("tool_call_id") not in code_interpreter_call_ids
    return False


def _strip_foreign_signature(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a block without another provider's thought signature.

    This runs before empty-block filtering so a signature-only foreign block does
    not become an invalid empty Gemini part after its signature is removed.
    """
    extras = block.get("extras")
    extras_has_signature = isinstance(extras, dict) and "signature" in extras
    if (
        not {"signature", "thought_signature"}.intersection(block)
        and not extras_has_signature
    ):
        return block
    stripped: dict[str, Any] = {
        k: v for k, v in block.items() if k not in ("signature", "thought_signature")
    }
    if extras_has_signature:
        stripped["extras"] = {
            k: v for k, v in cast("dict[str, Any]", extras).items() if k != "signature"
        }
    return stripped


def _is_empty_content_block(block: Mapping[str, Any]) -> bool:
    """Check whether a text or thought block has no replayable payload."""
    if block.get("type") not in ("text", "thinking", "reasoning"):
        return False
    has_text = (
        block.get("text")
        or block.get("thinking")
        or block.get("reasoning")
        or block.get("summary")
    )
    return not has_text and not _block_signature(block)


def _content_block_file_uri(block: Any) -> Any:
    """Return the URI carried by any supported file block shape."""
    if not isinstance(block, Mapping):
        return None
    file_data = block.get("file_data")
    if isinstance(file_data, Mapping):
        return file_data.get("file_uri")
    block_type = block.get("type")
    if block_type in ("file_data", "media"):
        return block.get("file_uri")
    if block_type in ("file", "video", "audio"):
        return block.get("file_id") or block.get("url")
    if block_type == "image":
        return block.get("url")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, Mapping):
            return image_url.get("url")
        return image_url
    return None


def _drop_unsupported_file_references(
    content: list[Any], *, use_vertexai: bool
) -> list[Any]:
    """Drop file references unsupported by the target backend."""
    filtered: list[Any] = []
    for block in content:
        file_uri = _content_block_file_uri(block)
        if not _is_file_uri_supported(file_uri, use_vertexai=use_vertexai):
            logger.warning(
                "Dropping Google Cloud Storage file URI from chat history because "
                "the target Gemini Developer API does not support gs:// references."
            )
            continue
        filtered.append(block)
    return filtered


def _prepare_ai_message_content(
    message: AIMessage, *, exclude_function_calls: bool
) -> list[Any]:
    """Normalize non-v1 AI content for provider-aware conversion.

    Foreign messages use LangChain's standardized blocks. Native messages retain
    their raw blocks because core may wrap valid Gemini media as `non_standard`.
    """
    model_provider = message.response_metadata.get("model_provider")
    provider_kind = _classify_model_provider(model_provider)
    is_foreign = provider_kind == "foreign"
    # Core standardizes foreign blocks, but can wrap native Gemini blocks as
    # `non_standard`, so native and unstamped messages use their raw content.
    content = message.content_blocks if is_foreign else message.content

    if not isinstance(content, list):
        return [content] if content else []

    code_interpreter_call_ids: set[str] = set()
    if is_foreign:
        code_interpreter_call_ids = {
            call_id
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "server_tool_call"
            and block.get("name") == "code_interpreter"
            and isinstance(call_id := block.get("id"), str)
        }

    filtered: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str) and not block:
                continue
            filtered.append(block)
            continue

        if exclude_function_calls and _is_rebuilt_tool_call_block(block):
            continue
        block_type = block.get("type")
        if is_foreign and block_type == "non_standard":
            value = block.get("value")
            logger.warning(
                "Dropping non-standard content block that cannot be represented "
                "as a Gemini part (inner type: %s).",
                value.get("type") if isinstance(value, Mapping) else None,
            )
            continue
        if exclude_function_calls and block_type == "function_call":
            continue
        if block_type == "file_data" and "file_uri" in block:
            filtered.append(_as_gemini_media_block(block))
            continue
        candidate: Mapping[str, Any] = block
        if is_foreign:
            if block_type in _REASONING_BLOCK_TYPES:
                # Mirrors the v1 conversion in `_compat.py`: stripping the foreign
                # signature is not enough, because the reasoning text itself is
                # another provider's chain-of-thought and is not a Gemini thought.
                logger.warning(
                    "Dropping reasoning block from provider %r; foreign reasoning "
                    "is not replayed as a Gemini thought part.",
                    model_provider,
                )
                continue
            if _is_unsupported_foreign_server_tool(block, code_interpreter_call_ids):
                continue
            candidate = _strip_foreign_signature(block)
        if _is_empty_content_block(candidate):
            continue
        filtered.append(candidate)
    return filtered


def _convert_ai_message_content(
    message: AIMessage,
    model: str | None,
    *,
    exclude_function_calls: bool,
    use_vertexai: bool,
) -> list[Part]:
    """Convert AI content while treating `message.tool_calls` as authoritative.

    Native content is validated strictly; foreign and provider-less content is
    converted leniently so unsupported provider-specific blocks do not abort replay.
    """
    provider_kind = _classify_model_provider(
        message.response_metadata.get("model_provider")
    )
    is_v1_content = message.response_metadata.get("output_version") in (
        "v1",
        "interactions/v1",
    )
    if is_v1_content:
        content: list[Any] = _convert_from_v1_to_generativelanguage_v1beta(
            cast("list[types.ContentBlock]", message.content),
            message.response_metadata.get("model_provider"),
            use_vertexai=use_vertexai,
        )
        if exclude_function_calls:
            content = [
                block for block in content if not _is_redundant_v1beta_part(block)
            ]
    elif exclude_function_calls or provider_kind == "foreign":
        content = _prepare_ai_message_content(
            message, exclude_function_calls=exclude_function_calls
        )
    else:
        content = (
            [message.content] if isinstance(message.content, str) else message.content
        )

    content = _drop_unsupported_file_references(content, use_vertexai=use_vertexai)
    convert = (
        _convert_to_parts if provider_kind == "native" else _convert_to_parts_lenient
    )
    return convert(
        content,
        model=model,
        allow_v1beta_dicts=is_v1_content,
    )


def _parse_chat_history(
    input_messages: Sequence[BaseMessage],
    convert_system_message_to_human: bool = False,
    model: str | None = None,
    *,
    use_vertexai: bool = False,
) -> tuple[Content | None, list[Content]]:
    """Parses sequence of `BaseMessage` into system instruction and formatted messages.

    Args:
        input_messages: Sequence of `BaseMessage` objects representing the chat history.
        convert_system_message_to_human: Deprecated, use system instructions instead.

            Whether to convert the first system message into a `HumanMessage`.
        model: The model name, used for version-specific logic.
        use_vertexai: Whether the target is the Vertex AI backend.

    Returns:
        A tuple containing:

            - An optional `generativelanguage_v1beta` `Content` representing the system
                instruction (if any).
            - A list of `generativelanguage_v1beta` `Content` representing the formatted
                messages.
    """
    if convert_system_message_to_human:
        warnings.warn(
            "The 'convert_system_message_to_human' parameter is deprecated and will be "
            "removed in a future version. Use system instructions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    formatted_messages: list[Content] = []

    system_instruction: Content | None = None
    messages_without_tool_messages = [
        message for message in input_messages if not isinstance(message, ToolMessage)
    ]
    tool_messages = [
        message for message in input_messages if isinstance(message, ToolMessage)
    ]
    for i, message in enumerate(messages_without_tool_messages):
        if isinstance(message, SystemMessage):
            system_parts = _convert_to_parts(message.content, model=model)
            if i == 0:
                system_instruction = Content(parts=system_parts)
            elif system_instruction is not None:
                if system_instruction.parts is None:
                    system_instruction.parts = system_parts
                else:
                    system_instruction.parts.extend(system_parts)
            else:
                pass
            continue
        if isinstance(message, AIMessage):
            role = "model"
            if message.tool_calls:
                # First, convert all non-function-call content (text, thinking,
                # reasoning, media, etc.) through the unified conversion path.
                # When include_thoughts=True, thinking blocks need to be preserved
                # when passing messages back to the API.
                ai_message_parts = _convert_ai_message_content(
                    message,
                    model,
                    exclude_function_calls=True,
                    use_vertexai=use_vertexai,
                )

                # Then, add function call parts
                function_call_sigs: dict[Any, str | bytes] = (
                    message.additional_kwargs.get(
                        _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
                    )
                )
                indexed_function_call_sigs = _function_call_signatures_from_content(
                    message
                )
                for tool_call_index, tool_call in enumerate(message.tool_calls):
                    function_call = FunctionCall(
                        name=tool_call["name"],
                        args=tool_call["args"],
                    )
                    # Check if there's a signature for this function call
                    sig = function_call_sigs.get(
                        tool_call.get("id")
                    ) or indexed_function_call_sigs.get(tool_call_index)
                    if sig:
                        ai_message_parts.append(
                            Part(
                                function_call=function_call,
                                thought_signature=_decode_signature(sig),
                            )
                        )
                    else:
                        ai_message_parts.append(Part(function_call=function_call))
                tool_messages_parts = _get_ai_message_tool_messages_parts(
                    tool_messages=tool_messages, ai_message=message, model=model
                )
                formatted_messages.append(Content(role=role, parts=ai_message_parts))
                # Only append tool response message if there are actual tool responses.
                # The Gemini API requires every Content message to have at least one
                # Part. If there are no ToolMessages in the conversation history for
                # this AI message's tool_calls, tool_messages_parts will be empty, and
                # we must not create an empty Content message.
                if tool_messages_parts:
                    formatted_messages.append(
                        Content(role="user", parts=tool_messages_parts)
                    )
                continue
            if raw_function_call := message.additional_kwargs.get("function_call"):
                function_call = FunctionCall(
                    name=raw_function_call["name"],
                    args=json.loads(raw_function_call["arguments"]),
                )
                parts = [Part(function_call=function_call)]
            elif message.response_metadata.get("output_version") == "v1":
                parts = _convert_ai_message_content(
                    message,
                    model,
                    exclude_function_calls=False,
                    use_vertexai=use_vertexai,
                )
            else:
                parts = _convert_ai_message_content(
                    message,
                    model,
                    exclude_function_calls=False,
                    use_vertexai=use_vertexai,
                )
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message.content, model=model)
            if i == 1 and convert_system_message_to_human and system_instruction:
                parts = list(system_instruction.parts or []) + parts
                system_instruction = None
        elif isinstance(message, FunctionMessage):
            role = "user"
            parts = _convert_tool_message_to_parts(message, model=model)
        else:
            msg = f"Unexpected message with type {type(message)} at the position {i}."
            raise ValueError(msg)

        # Final step; assemble the Content object to pass to the API
        # If version = "v1", the parts are already in v1beta format and will be
        # automatically converted using protobuf's auto-conversion
        if role == "model" and not parts:
            if isinstance(message.content, list) and message.content:
                logger.warning(
                    "AI message at index %d converted to no Gemini parts after all "
                    "%d content block(s) were dropped; using an empty text part.",
                    i,
                    len(message.content),
                )
            else:
                logger.warning(
                    "AI message at index %d converted to no Gemini parts; using an "
                    "empty text part.",
                    i,
                )
            parts = [Part(text="")]
        formatted_messages.append(Content(role=role, parts=parts))

    # Enforce thought signatures for new Gemini models
    #
    # These models require a 'thought_signature' field in function calls for the
    # current active conversation loop. If missing (e.g., from older history or
    # manual construction), the API may reject the request.
    if model and _is_gemini_3_or_later(model):
        # 1. Identify the "Active Loop":
        # Scan backwards to find the most recent User message that initiated he current
        # interaction (i.e., contains text/media, not just a tool response).
        # This defines the scope where we must ensure compliance.
        active_loop_start_idx = -1
        for i in range(len(formatted_messages) - 1, -1, -1):
            content_msg = formatted_messages[i]
            if content_msg.role == "user":
                has_function_response = False
                has_standard_content = False
                for part in content_msg.parts or []:
                    if part.function_response:
                        has_function_response = True
                    if part.text or part.inline_data:
                        has_standard_content = True

                # Found the user message that started this turn
                if has_standard_content and not has_function_response:
                    active_loop_start_idx = i
                    break

        # 2. Patch Missing Signatures:
        # Iterate through the active loop. If a model message contains a function call
        # but lacks a thought signature, inject the bypass sentinel. This satisfies
        # the API's validation without requiring the original internal thought data.
        start_idx = active_loop_start_idx + 1 if active_loop_start_idx != -1 else 0
        for i in range(start_idx, len(formatted_messages)):
            content_msg = formatted_messages[i]
            if content_msg.role == "model":
                first_fc_seen = False
                for part in content_msg.parts or []:
                    if part.function_call:
                        if not first_fc_seen:
                            if not part.thought_signature:
                                # Assign the str sentinel post-construction so
                                # pydantic never coerces it to bytes: the SDK's
                                # request encoder base64-encodes bytes values,
                                # which would hide the bypass string from the
                                # API. Part is not validate_assignment, so the
                                # string survives serialization as-is.
                                try:
                                    part.thought_signature = cast(
                                        "bytes",
                                        SKIP_THOUGHT_SIGNATURE_VALIDATOR,
                                    )
                                except ValueError:
                                    # A future SDK/pydantic version could turn
                                    # on validate_assignment, which coerces the
                                    # str to base64-decoded bytes. Send those
                                    # bytes rather than dropping the signature
                                    # entirely and failing schema validation.
                                    part.thought_signature = DUMMY_THOUGHT_SIGNATURE
                            first_fc_seen = True

    return system_instruction, formatted_messages


# Helper function to append content consistently
def _append_to_content(
    current_content: str | list[Any] | None, new_item: Any
) -> str | list[Any]:
    """Appends a new item to the content, handling different initial content types."""
    if current_content is None and isinstance(new_item, str):
        return new_item
    if current_content is None:
        return [new_item]
    if isinstance(current_content, str):
        return [current_content, new_item]
    if isinstance(current_content, list):
        current_content.append(new_item)
        return current_content
    # This case should ideally not be reached with proper type checking,
    # but it catches any unexpected types that might slip through.
    msg = f"Unexpected content type: {type(current_content)}"
    raise TypeError(msg)


#: Block types that arrive as incremental deltas and must keep a stable index so
#: successive chunks merge into one block. Every other block Gemini emits is
#: complete on arrival and needs an index of its own.
_MERGEABLE_BLOCK_TYPES = frozenset({"text", "thinking"})


class _StreamBlockIndexer:
    """Allocates streaming `index` values for content blocks."""

    def __init__(self) -> None:
        self._next_index = 0
        self._open_type: str | None = None
        self._open_index = -1

    def _allocate(self, block_type: str | None) -> int:
        index = self._next_index
        self._next_index += 1
        self._open_type = block_type
        self._open_index = index
        return index

    def assign(self, message: AIMessageChunk) -> None:
        """Populate any missing ``index`` fields on ``message``, in place.

        Args:
            message: The chunk whose content blocks and tool call chunks should
                be indexed. Existing ``index`` values are left untouched.
        """
        if isinstance(message.content, list):
            for block in message.content:
                if not isinstance(block, dict) or "type" not in block:
                    continue
                block_type = block["type"]
                if (
                    block_type == self._open_type
                    and block_type in _MERGEABLE_BLOCK_TYPES
                ):
                    index = self._open_index
                else:
                    index = self._allocate(block_type)
                if "index" not in block:
                    block["index"] = index

        for tool_call_chunk_ in message.tool_call_chunks:
            # Gemini emits each function call complete in a single part, so every
            # chunk without a provider-supplied index is a distinct tool call.
            if tool_call_chunk_.get("index") is None:
                tool_call_chunk_["index"] = self._allocate(None)


def _convert_integer_like_floats(obj: Any) -> Any:
    """Convert integer-like floats to integers recursively.

    Addresses a protobuf issue where integers are converted to floats when using
    `proto.Message.to_dict()`.

    Args:
        obj: The object to process (can be `dict`, `list`, or primitive)

    Returns:
        The object with integer-like floats converted to integers
    """
    if isinstance(obj, dict):
        return {k: _convert_integer_like_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_integer_like_floats(item) for item in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def _parse_response_candidate(
    response_candidate: Candidate,
    streaming: bool = False,
    model_name: str | None = None,
    *,
    model_name_for_content: str | None = None,
) -> AIMessage:
    """Parse a response candidate from Google into an `AIMessage`.

    Args:
        response_candidate: The candidate from the API response.
        streaming: Whether this is a streaming response.
        model_name: Model name to include in `response_metadata` (`None` for
            intermediate streaming chunks to avoid duplication when concatenating).
        model_name_for_content: Model name to use for determining content format.

            For Gemini 3+, this determines whether to use list-based content blocks
            or string content. Must be consistent across all streaming chunks to
            enable proper concatenation. If not provided, falls back to `model_name`.

    Returns:
        The parsed `AIMessage` or `AIMessageChunk`.

    Note:
        During streaming, we want to avoid duplicating `model_name` in
        `response_metadata` for intermediate chunks (only include it in the final
        chunk), but we need the model name consistently across all chunks to determine
        the content format. This is why `model_name` and `model_name_for_content` are
        separate parameters.
    """
    content: str | list[str | dict] | None = None
    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {"model_provider": "google_genai"}
    if model_name:
        response_metadata["model_name"] = model_name
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    # Use model_name_for_content if provided, otherwise fall back to model_name.
    # This ensures consistent content format across all streaming chunks while
    # only including model_name in response_metadata for the final chunk.
    effective_model_name = model_name_for_content or model_name

    parts = response_candidate.content.parts or [] if response_candidate.content else []
    for i, part in enumerate(parts):
        text: str | None = None
        try:
            if hasattr(part, "text") and part.text is not None:
                text = part.text
                # Remove erroneous newline character if present
                if not streaming:
                    text = text.rstrip("\n")
        except AttributeError:
            pass

        # Extract thought signature if present (can be on any Part type)
        # Signatures are binary data, encode to base64 string for JSON serialization
        thought_sig: str | None = None
        if hasattr(part, "thought_signature") and part.thought_signature:
            try:
                # Encode binary signature to base64 string
                thought_sig = base64.b64encode(part.thought_signature).decode("ascii")
                if not thought_sig:  # Empty string
                    thought_sig = None
            except (AttributeError, TypeError):
                thought_sig = None

        if hasattr(part, "thought") and part.thought:
            thinking_message = {
                "type": "thinking",
                "thinking": part.text,
            }
            # Include signature if present
            if thought_sig:
                thinking_message["signature"] = thought_sig
            content = _append_to_content(content, thinking_message)
        elif (
            (text is not None and text)  # text part with non-empty string
            or (part.text is not None and thought_sig)  # text part w/ thought sig
        ):
            text_block: dict[str, Any] = {"type": "text", "text": text or ""}
            if thought_sig:
                text_block["extras"] = {"signature": thought_sig}
            if thought_sig or _is_gemini_3_or_later(effective_model_name or ""):
                # append blocks if there's a signature or new Gemini model
                content = _append_to_content(content, text_block)
            elif isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") == "thinking"
                for item in content
            ):
                # if there's thinking blocks, keep content as dicts
                content = _append_to_content(content, text_block)
            else:
                # otherwise, append text
                content = _append_to_content(content, text or "")

        if hasattr(part, "executable_code") and part.executable_code is not None:
            if part.executable_code.code and part.executable_code.language:
                code_id = str(uuid.uuid4())  # Generate ID if not present, needed later
                code_message = {
                    "type": "executable_code",
                    "executable_code": part.executable_code.code,
                    "language": part.executable_code.language,
                    "id": code_id,
                }
                content = _append_to_content(content, code_message)

        if (
            hasattr(part, "code_execution_result")
            and part.code_execution_result is not None
        ) and part.code_execution_result.output:
            # outcome: 1 = OUTCOME_OK (success), 2 = error
            # Convert enum to int for compatibility with langchain_core
            if part.code_execution_result.outcome is None:
                outcome = 1  # Default to OUTCOME_OK
            elif (
                part.code_execution_result.outcome
                == CodeExecutionResultOutcome.OUTCOME_OK
            ):
                outcome = 1
            else:
                outcome = 2
            execution_result = {
                "type": "code_execution_result",
                "code_execution_result": part.code_execution_result.output,
                "outcome": outcome,
                "tool_call_id": "",  # Linked via block translator
            }
            content = _append_to_content(content, execution_result)

        # Server-side media processing steps (e.g., agentic video understanding).
        media_call = getattr(part, "tool_call", None)
        if media_call is not None:
            media_call_block: dict[str, Any] = {
                "type": "server_tool_call",
                "name": _server_tool_name(media_call.tool_type),
                "id": media_call.id or str(uuid.uuid4()),
                "args": dict(media_call.args or {}),
            }
            if thought_sig:
                media_call_block["extras"] = {"signature": thought_sig}
            content = _append_to_content(content, media_call_block)

        media_result = getattr(part, "tool_response", None)
        if media_result is not None:
            media_result_block: dict[str, Any] = {
                "type": "server_tool_result",
                "tool_call_id": media_result.id or "",
                "status": "success",
                "output": dict(media_result.response or {}),
                # `ServerToolResult` has no `name`, so the originating tool is
                # recorded here, mirroring how code execution results are tagged.
                "extras": {"block_type": _server_tool_name(media_result.tool_type)},
            }
            if thought_sig:
                media_result_block["extras"]["signature"] = thought_sig
            content = _append_to_content(content, media_result_block)

        if part.inline_data and part.inline_data.data and part.inline_data.mime_type:
            if part.inline_data.mime_type.startswith("audio/"):
                buffer = io.BytesIO()

                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    # TODO: Read Sample Rate from MIME content type.
                    wf.setframerate(24000)
                    wf.writeframes(part.inline_data.data)

                audio_data = buffer.getvalue()
                additional_kwargs["audio"] = audio_data

                # For backwards compatibility, audio stays in additional_kwargs by
                # default and is accessible via .content_blocks property

            if part.inline_data.mime_type.startswith("image/"):
                image_format = part.inline_data.mime_type[6:]
                image_message: dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_bytes_to_b64_string(
                            part.inline_data.data,
                            image_format=image_format,
                        )
                    },
                }
                if thought_sig:
                    image_message["extras"] = {"signature": thought_sig}
                content = _append_to_content(content, image_message)

        if part.function_call:
            function_call = {"name": part.function_call.name}
            # dump to match other function calling llm for now
            # Convert function call args to dict first, then fix integer-like floats
            args_dict = dict(part.function_call.args) if part.function_call.args else {}
            function_call_args_dict = _convert_integer_like_floats(args_dict)
            function_call["arguments"] = json.dumps(
                {k: function_call_args_dict[k] for k in function_call_args_dict}
            )
            additional_kwargs["function_call"] = function_call

            raw_id = getattr(part.function_call, "id", None)
            tool_call_id = str(raw_id) if raw_id else str(uuid.uuid4())
            if streaming:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=tool_call_id,
                        index=function_call.get("index"),  # type: ignore
                    )
                )
            else:
                try:
                    tool_call_dict = parse_tool_calls(
                        [{"function": function_call}],
                        return_id=False,
                    )[0]
                except Exception as e:
                    invalid_tool_calls.append(
                        invalid_tool_call(
                            name=function_call.get("name"),
                            args=function_call.get("arguments"),
                            id=tool_call_id,
                            error=str(e),
                        )
                    )
                else:
                    tool_calls.append(
                        tool_call(
                            name=tool_call_dict["name"],
                            args=tool_call_dict["args"],
                            id=tool_call_id,
                        )
                    )

            # If this function_call Part has a signature, track it separately
            if thought_sig:
                if _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY not in additional_kwargs:
                    additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY] = {}
                additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY][
                    tool_call_id
                ] = (
                    _bytes_to_base64(thought_sig)
                    if isinstance(thought_sig, bytes)
                    else thought_sig
                )

    if content is None:
        if _is_gemini_3_or_later(effective_model_name or ""):
            content = []
        else:
            content = ""
    if isinstance(content, list) and any(
        isinstance(item, dict) and "executable_code" in item for item in content
    ):
        warnings.warn(
            """
        Warning: Output may vary each run.
        - 'executable_code': Always present.
        - 'execution_result' & 'image_url': May be absent for some queries.

        Validate before using in production.
"""
        )
    if streaming:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


def _finish_reason_fields(
    finish_reason: str | None, *, has_tool_calls: bool
) -> dict[str, str]:
    """Map a Gemini `finish_reason` onto `normalized_finish_reason`."""
    if not finish_reason:
        return {}
    normalized = _FINISH_REASONS.get(finish_reason.upper(), "other")
    # Gemini reports `STOP` when it ends on function calls.
    if normalized == "stop" and has_tool_calls:
        normalized = "tool_call"
    return {"normalized_finish_reason": normalized}


def _mark_streamed_tool_call(gen: ChatGenerationChunk, saw_tool_call: bool) -> bool:
    """Upgrade a final `stop` to `tool_call` when an earlier chunk had tool calls.

    Gemini can send function calls and `STOP` in separate stream chunks.
    """
    message = gen.message
    if isinstance(message, AIMessageChunk) and (
        message.tool_call_chunks or message.tool_calls
    ):
        saw_tool_call = True
    info = gen.generation_info
    if saw_tool_call and info and info.get("normalized_finish_reason") == "stop":
        info["normalized_finish_reason"] = "tool_call"
    return saw_tool_call


def _response_to_result(
    response: GenerateContentResponse,
    stream: bool = False,
    prev_usage: UsageMetadata | None = None,
) -> ChatResult:
    """Converts a Google AI response into a LangChain `ChatResult`."""
    llm_output = (
        {"prompt_feedback": response.prompt_feedback.model_dump()}
        if response.prompt_feedback
        else {}
    )

    # Get usage metadata
    try:
        if response.usage_metadata is None:
            msg = "Usage metadata is None"
            raise AttributeError(msg)
        # Server-side tool use (for example agentic video processing) reports its
        # prompt tokens separately; they are input tokens and are included in
        # `total_token_count`, so fold them in to keep the parts summing.
        input_tokens = (response.usage_metadata.prompt_token_count or 0) + (
            response.usage_metadata.tool_use_prompt_token_count or 0
        )
        thought_tokens = response.usage_metadata.thoughts_token_count or 0
        output_tokens = (
            response.usage_metadata.candidates_token_count or 0
        ) + thought_tokens
        total_tokens = response.usage_metadata.total_token_count or 0
        # Cached tokens are a subset of `prompt_token_count`. With grounding (for
        # example `google_search`), Gemini can report a cached count well above
        # the prompt. That count can't be trusted, and clamping it would report a
        # full cache hit, so drop it instead.
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        cache_read_tokens = response.usage_metadata.cached_content_token_count or 0
        if cache_read_tokens > prompt_tokens:
            cache_read_tokens = 0
        if input_tokens + output_tokens + cache_read_tokens + total_tokens > 0:
            if thought_tokens > 0:
                cumulative_usage = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details={"cache_read": cache_read_tokens},
                    output_token_details={"reasoning": thought_tokens},
                )
            else:
                cumulative_usage = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details={"cache_read": cache_read_tokens},
                )
            # previous usage metadata needs to be subtracted because gemini api returns
            # already-accumulated token counts with each chunk
            lc_usage = subtract_usage(cumulative_usage, prev_usage)
            if prev_usage and cumulative_usage["input_tokens"] < prev_usage.get(
                "input_tokens", 0
            ):
                # Gemini 2.0 returns a lower cumulative count of prompt tokens
                # in the final chunk. We take this count to be ground truth because
                # it's consistent with the reported total tokens. So we need to
                # ensure this chunk compensates (the subtract_usage funcction floors
                # at zero).
                lc_usage["input_tokens"] = cumulative_usage[
                    "input_tokens"
                ] - prev_usage.get("input_tokens", 0)
        else:
            lc_usage = None
    except AttributeError:
        lc_usage = None

    generations: list[ChatGeneration] = []

    for candidate in response.candidates or []:
        generation_info: dict[str, Any] = {}
        # Only include model_name in response_metadata for the last chunk
        # (when finish_reason exists)
        # to avoid duplication when chunks are concatenated with +=
        model_name_for_metadata = None
        if candidate.finish_reason:
            # Handle finish_reason that may be an enum or raw integer
            if hasattr(candidate.finish_reason, "name"):
                generation_info["finish_reason"] = candidate.finish_reason.name
            elif isinstance(candidate.finish_reason, int):
                generation_info["finish_reason"] = f"UNKNOWN_{candidate.finish_reason}"
            # Add model_name in last chunk
            generation_info["model_name"] = response.model_version or ""
            # Set for final chunk
            model_name_for_metadata = response.model_version
        generation_info["safety_ratings"] = (
            [safety_rating.model_dump() for safety_rating in candidate.safety_ratings]
            if candidate.safety_ratings
            else []
        )
        # Pass model_version for content format determination (Gemini 3+ needs
        # consistent list-based content across all chunks), but only include
        # model_name in response_metadata for the final chunk to avoid duplication
        # when chunks are concatenated.
        message = _parse_response_candidate(
            candidate,
            streaming=stream,
            model_name=model_name_for_metadata,  # None for intermediate chunks
            model_name_for_content=response.model_version,  # Always set
        )

        if not hasattr(message, "response_metadata"):
            message.response_metadata = {}

        generation_info.update(
            _finish_reason_fields(
                generation_info.get("finish_reason"),
                has_tool_calls=bool(message.tool_calls or message.invalid_tool_calls),
            )
        )

        if generation_info.get("finish_reason") in _REFUSAL_FINISH_REASONS:
            # `v1` content is built from the message before `generation_info` is merged.
            message.response_metadata["finish_reason"] = generation_info[
                "finish_reason"
            ]
            if candidate.finish_message:
                message.response_metadata["finish_message"] = candidate.finish_message

        try:
            if candidate.grounding_metadata:
                grounding_metadata = candidate.grounding_metadata.model_dump()
                # Ensure None fields that are expected to be lists become empty lists
                # to prevent errors in downstream processing
                if (
                    "grounding_supports" in grounding_metadata
                    and grounding_metadata["grounding_supports"] is None
                ):
                    grounding_metadata["grounding_supports"] = []
                if (
                    "grounding_chunks" in grounding_metadata
                    and grounding_metadata["grounding_chunks"] is None
                ):
                    grounding_metadata["grounding_chunks"] = []
                if (
                    "web_search_queries" in grounding_metadata
                    and grounding_metadata["web_search_queries"] is None
                ):
                    grounding_metadata["web_search_queries"] = []
                if (
                    "image_search_queries" in grounding_metadata
                    and grounding_metadata["image_search_queries"] is None
                ):
                    grounding_metadata["image_search_queries"] = []
                generation_info["grounding_metadata"] = grounding_metadata
                message.response_metadata["grounding_metadata"] = grounding_metadata
        except AttributeError:
            pass

        message.usage_metadata = lc_usage

        if stream:
            generations.append(
                ChatGenerationChunk(
                    message=cast("AIMessageChunk", message),
                    generation_info=generation_info,
                )
            )
        else:
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
    if not response.candidates:
        # Likely a "prompt feedback" violation (e.g., toxic input)
        # Raising an error would be different than how OpenAI handles it,
        # so we'll just log a warning and continue with an empty message.
        logger.warning(
            "Gemini produced an empty response. Continuing with empty message\n"
            f"Feedback: {response.prompt_feedback}"
        )
        response_metadata: dict[str, Any] = {"model_provider": "google_genai"}
        if response.prompt_feedback:
            response_metadata["prompt_feedback"] = response.prompt_feedback.model_dump()
        if stream:
            generations = [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata=response_metadata,
                        usage_metadata=lc_usage,
                    ),
                    generation_info={},
                )
            ]
        else:
            generations = [
                ChatGeneration(
                    message=AIMessage("", response_metadata=response_metadata),
                    generation_info={},
                )
            ]
    return ChatResult(generations=generations, llm_output=llm_output)


class ChatGoogleGenerativeAI(_BaseGoogleGenerativeAI, BaseChatModel):
    r"""Google GenAI chat model integration.

    Setup:
        !!! version-added "Vertex AI Platform Support"

            Added in `langchain-google-genai` 4.0.0.

            `ChatGoogleGenerativeAI` now supports both the **Gemini Developer API** and
            **Vertex AI Platform** as backend options.

        **For Gemini Developer API** (simplest):

        1. Set the `GOOGLE_API_KEY` environment variable (recommended), or
        2. Pass your API key using the `api_key` parameter

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", api_key="...")
        ```

        **For Vertex AI Platform with API key**:

        ```bash
        export GEMINI_API_KEY='your-api-key'
        export GOOGLE_GENAI_USE_VERTEXAI=true
        export GOOGLE_CLOUD_PROJECT='your-project-id'
        ```

        ```python
        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
        # Or explicitly:
        model = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            api_key="...",
            project="your-project-id",
            vertexai=True,
        )
        ```

        **For Vertex AI with credentials**:

        ```python
        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            project="your-project-id",
            # Uses Application Default Credentials (ADC)
        )
        ```

        **Automatic backend detection** (when `vertexai=None` / unspecified):

        1. If `GOOGLE_GENAI_USE_VERTEXAI` env var is set, uses that value
        2. If `credentials` parameter is provided, uses Vertex AI
        3. If `project` parameter is provided, uses Vertex AI
        4. Otherwise, uses Gemini Developer API

    Environment variables:
        | Variable | Purpose | Backend |
        |----------|---------|---------|
        | `GOOGLE_API_KEY` | API key (primary) | Both (see `GOOGLE_GENAI_USE_VERTEXAI`) |
        | `GEMINI_API_KEY` | API key (fallback) | Both (see `GOOGLE_GENAI_USE_VERTEXAI`) |
        | `GOOGLE_GENAI_USE_VERTEXAI` | Force Vertex AI backend (`true`/`false`) | Vertex AI |
        | `GOOGLE_CLOUD_PROJECT` | GCP project ID | Vertex AI |
        | `GOOGLE_CLOUD_LOCATION` | GCP region (default: `global`) | Vertex AI |
        | `HTTPS_PROXY` | HTTP/HTTPS proxy URL | Both |
        | `SSL_CERT_FILE` | Custom SSL certificate file | Both |

        `GOOGLE_API_KEY` is checked first for backwards compatibility. (`GEMINI_API_KEY`
        was introduced later to better reflect the API's branding.)

    Proxy configuration:
        Set these before initializing:

        ```bash
        export HTTPS_PROXY='http://username:password@proxy_uri:port'
        export SSL_CERT_FILE='path/to/cert.pem'  # Optional: custom SSL certificate
        ```

        For SOCKS5 proxies or advanced proxy configuration, use the
        `client_args` parameter:

        ```python
        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            client_args={"proxy": "socks5://user:pass@host:port"},
        )
        ```

    ???+ example "Instantiation"

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
        model.invoke("Write me a ballad about LangChain")
        ```

    ???+ example "Invoke"

        ```python
        messages = [
            ("system", "Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

        ```python
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "**J'adore la programmation.**\n\nYou can also say:...",
                    "extras": {"signature": "Eq0W..."},
                }
            ],
            additional_kwargs={},
            response_metadata={
                "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
                "finish_reason": "STOP",
                "model_name": "gemini-3.1-pro-preview",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--63a04ced-6b63-4cf6-86a1-c32fa565938e-0",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 826,
                "total_tokens": 838,
                "input_token_details": {"cache_read": 0},
                "output_token_details": {"reasoning": 777},
            },
        )
        ```

        !!! note "`content` format"

            The shape of `content` may differ based on the model chosen. See
            [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#invocation)
            for more info.

    ???+ example "Stream"

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

        for chunk in model.stream(messages):
            print(chunk)
        ```

        ```python
        AIMessageChunk(
            content="J",
            response_metadata={"finish_reason": "STOP", "safety_ratings": []},
            id="run-e905f4f4-58cb-4a10-a960-448a2bb649e3",
            usage_metadata={
                "input_tokens": 18,
                "output_tokens": 1,
                "total_tokens": 19,
            },
        )
        AIMessageChunk(
            content="'adore programmer. \\n",
            response_metadata={
                "finish_reason": "STOP",
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                ],
            },
            id="run-e905f4f4-58cb-4a10-a960-448a2bb649e3",
            usage_metadata={
                "input_tokens": 18,
                "output_tokens": 5,
                "total_tokens": 23,
            },
        )
        ```

        To assemble a full [`AIMessage`][langchain.messages.AIMessage] message from a
        stream of chunks:

        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(
            content="J'adore programmer. \\n",
            response_metadata={
                "finish_reason": "STOPSTOP",
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "blocked": False,
                    },
                ],
            },
            id="run-3ce13a42-cd30-4ad7-a684-f1f0b37cdeec",
            usage_metadata={
                "input_tokens": 36,
                "output_tokens": 6,
                "total_tokens": 42,
            },
        )
        ```

        !!! note "`content` format"

            The shape of `content` may differ based on the model chosen. See
            [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#invocation)
            for more info.

    ???+ example "Async invocation"

        ```python
        await model.ainvoke(messages)

        # stream:
        async for chunk in (await model.astream(messages))

        # batch:
        await model.abatch([messages])
        ```

    ???+ example "Tool calling"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#tool-calling)
        for more info.

        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
        ai_msg = llm_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        ```python
        [
            {
                "name": "GetWeather",
                "args": {"location": "Los Angeles, CA"},
                "id": "c186c99f-f137-4d52-947f-9e3deabba6f6",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York City, NY"},
                "id": "cebd4a5d-e800-4fa5-babd-4aa286af4f31",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "4f92d897-f5e4-4d34-a3bc-93062c92591e",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York City, NY"},
                "id": "634582de-5186-4e4b-968b-f192f0a93678",
            },
        ]
        ```

    ???+ example "Structured output"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#structured-output)
        for more info.

        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: Optional[int] = Field(
                description="How funny the joke is, from 1 to 10"
            )


        # Default method uses json_schema for reliable structured output
        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")

        # Alternative: use function_calling method (less reliable)
        structured_model_fc = model.with_structured_output(
            Joke, method="function_calling"
        )
        ```

        ```python
        Joke(
            setup="Why are cats so good at video games?",
            punchline="They have nine lives on the internet",
            rating=None,
        )
        ```

        Two methods are supported for structured output:

        * `method='json_schema'` (default): Uses Gemini's native structured output API.

            The Google GenAI SDK automatically transforms schemas to ensure
            compatibility with Gemini. This includes:

            - Inlining `$defs` definitions (Union types work correctly)
            - Resolving `$ref` references for nested schemas
            - Property ordering preservation
            - Support for streaming partial JSON chunks

            Uses Gemini's `response_json_schema` API param. Refer to the Gemini API
            [docs](https://ai.google.dev/gemini-api/docs/structured-output) for more
            details. This method is recommended for better reliability as it
            constrains the model's generation process directly.

        * `method='function_calling'`: Uses tool calling to extract structured data.
            Less reliable than `json_schema` but compatible with all models.

    ???+ example "Image input"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#image-input)
        for more info.

        ```python
        import base64
        import httpx
        from langchain.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
        ai_msg = model.invoke([message])
        ai_msg.content
        ```

        ```txt
        The weather in this image appears to be sunny and pleasant. The sky is a bright
        blue with scattered white clouds, suggesting fair weather. The lush green grass
        and trees indicate a warm and possibly slightly breezy day. There are no...
        ```

    ???+ example "PDF input"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#pdf-input)
        for more info.

        ```python
        import base64
        from langchain.messages import HumanMessage

        pdf_bytes = open("/path/to/your/test.pdf", "rb").read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the document in a sentence"},
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": pdf_base64,
                },
            ]
        )
        ai_msg = model.invoke([message])
        ```

    ???+ example "Audio input"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#audio-input)
        for more info.

        ```python
        import base64
        from langchain.messages import HumanMessage

        audio_bytes = open("/path/to/your/audio.mp3", "rb").read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "summarize this audio in a sentence"},
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "audio/mp3",
                    "data": audio_base64,
                },
            ]
        )
        ai_msg = model.invoke([message])
        ```

    ???+ example "Video input"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#video-input)
        for more info.

        ```python
        import base64
        from langchain.messages import HumanMessage

        video_bytes = open("/path/to/your/video.mp4", "rb").read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "describe what's in this video in a sentence",
                },
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": "video/mp4",
                    "data": video_base64,
                },
            ]
        )
        ai_msg = model.invoke([message])
        ```

        You can also pass YouTube URLs directly:

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

        message = HumanMessage(
            content=[
                {
                    "type": "media",
                    "file_uri": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                },
                {"type": "text", "text": "Summarize the video in 3 sentences."},
            ]
        )
        response = model.invoke([message])
        print(response.text)
        ```

        Video inputs default to static processing, sampling frames at a fixed rate.
        On Gemini 3 and later, setting `media_processing` to `"AGENTIC"` instead
        lets the model navigate the timeline itself and load only the frames it
        needs, which is more token-efficient on long-form content. Note that
        `mime_type` is required whenever `media_processing` is set:

        ```python
        model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

        message = HumanMessage(
            content=[
                {
                    "type": "video",
                    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "mime_type": "video/mp4",
                    "media_processing": "AGENTIC",
                },
                {"type": "text", "text": "Summarize the video in 3 sentences."},
            ]
        )
        response = model.invoke([message])
        ```

        Videos uploaded through the Files API use `file_id`, and each video sets
        its own mode, so modes can be mixed in one request:

        ```python
        from google import genai

        client = genai.Client()
        lecture = client.files.upload(file="lecture.mp4")
        experiment = client.files.upload(file="experiment.mp4")
        # Wait for `client.files.get(name=...).state` to reach `ACTIVE` first.

        message = HumanMessage(
            content=[
                {
                    "type": "video",
                    "file_id": lecture.uri,
                    "mime_type": lecture.mime_type,
                    "media_processing": "AGENTIC",
                },
                {
                    "type": "video",
                    "file_id": experiment.uri,
                    "mime_type": experiment.mime_type,
                    "media_processing": "STATIC",
                },
                {"type": "text", "text": "Compare the lecture with the experiment."},
            ]
        )
        response = model.invoke([message])
        ```

        Inline base64 also works, subject to the usual request size limit:

        ```python
        import base64
        from pathlib import Path

        message = HumanMessage(
            content=[
                {
                    "type": "video",
                    "base64": base64.b64encode(Path("clip.mp4").read_bytes()).decode(),
                    "mime_type": "video/mp4",
                    "media_processing": "AGENTIC",
                },
                {"type": "text", "text": "What happens at 00:11?"},
            ]
        )
        response = model.invoke([message])
        ```

    ???+ example "Image generation"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#image-generation)
        for more info.

    ???+ example "Audio generation"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#audio-generation)
        for more info.

        !!! note "Vertex compatibility"

            Audio generation models (TTS) are currently in preview on Vertex AI
            and may require allowlist access. If you receive an `INVALID_ARGUMENT`
            error when using TTS models with `vertexai=True`, your project may need to
            be allowlisted.

            See this post on the [Google AI forum](https://discuss.ai.google.dev/t/request-allowlist-access-for-audio-output-in-gemini-2-5-pro-flash-tts-vertex-ai/108067)
            for more details.

    ???+ example "File upload"

        You can also upload files to Google's servers and reference them by URI.

        This works for PDFs, images, videos, and audio files.

        ```python
        import time
        from google import genai
        from langchain.messages import HumanMessage

        client = genai.Client()

        myfile = client.files.upload(file="/path/to/your/sample.pdf")
        while myfile.state.name == "PROCESSING":
            time.sleep(2)
            myfile = client.files.get(name=myfile.name)

        message = HumanMessage(
            content=[
                {"type": "text", "text": "What is in the document?"},
                {
                    "type": "media",
                    "file_uri": myfile.uri,
                    "mime_type": "application/pdf",
                },
            ]
        )
        ai_msg = model.invoke([message])
        ```

    ???+ example "Thinking"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#thinking-support)
        for more info.

        Gemini 3+ models use [`thinking_level`][langchain_google_genai.ChatGoogleGenerativeAI.thinking_level]
        to control reasoning depth. Supported levels and defaults vary by model. Check
        `model.profile` for `reasoning_effort_levels` and
        `reasoning_effort_default`.

        ```python
        model = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            thinking_level="low",  # For faster, lower-latency responses
        )
        ```

        Gemini 2.5 models use `thinking_budget`
        (an integer token count) to control reasoning. Set to `0` to disable thinking
        (where supported), or `-1` for dynamic thinking.

        See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for
        more details on thinking models.

        To see a thinking model's thoughts, set `include_thoughts=True`
        to have the model's reasoning summaries included in the response.

        ```python
        model = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            include_thoughts=True,
        )
        ai_msg = model.invoke("How many 'r's are in the word 'strawberry'?")
        ```

    ???+ example "Thought signatures"

        Gemini 3+ models return *thought signatures*—encrypted representations of
        the model's internal reasoning.

        For multi-turn conversations involving tool calls, you must pass the full
        [`AIMessage`][langchain.messages.AIMessage] back to the model so that these
        signatures are preserved. This happens automatically when you append the
        [`AIMessage`][langchain.messages.AIMessage] to your message list.

        See the [LangChain docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#thought-signatures) for more info as well as a code example.

        See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking)
        for more details on thought signatures.

    ???+ example "Google search"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#google-search)
        for more info.

        ```python
        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
        response = model.invoke(
            "When is the next total solar eclipse in US?",
            tools=[{"google_search": {}}],
        )
        response.content_blocks
        ```

        Alternatively, you can bind the tool to the model for easier reuse across calls:

        ```python
        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

        model_with_search = model.bind_tools([{"google_search": {}}])
        response = model_with_search.invoke(
            "When is the next total solar eclipse in US?"
        )

        response.content_blocks
        ```

    ???+ example "Google Maps"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#google-maps)
        for more info.

    ???+ example "Code execution"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#code-execution)
        for more info.

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

        model_with_code_interpreter = model.bind_tools([{"code_execution": {}}])
        response = model_with_code_interpreter.invoke("Use Python to calculate 3^3.")

        response.content_blocks
        ```

        ```output
        [{'type': 'server_tool_call',
          'name': 'code_interpreter',
          'args': {'code': 'print(3**3)', 'language': <Language.PYTHON: 1>},
          'id': '...'},
         {'type': 'server_tool_result',
          'tool_call_id': '',
          'status': 'success',
          'output': '27\n',
          'extras': {'block_type': 'code_execution_result',
           'outcome': 1}},
         {'type': 'text', 'text': 'The calculation of 3 to the power of 3 is 27.'}]
        ```

    ???+ example "Computer use"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#computer-use)
        for more info.

        !!! warning "Preview model limitations"

            The Computer Use model is in preview and may produce unexpected behavior.

            Always supervise automated tasks and avoid use with sensitive data or
            critical operations. See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/computer-use)
            for safety best practices.

    ???+ example "Token usage"
        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#token-usage-tracking)
        for more info.

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 18, "output_tokens": 5, "total_tokens": 23}
        ```

    ???+ example "Safety settings"

        Gemini models have default safety settings that can be overridden. If you
        are receiving lots of "Safety Warnings" from your models, you can try
        tweaking the `safety_settings` attribute of the model. For example, to
        turn off safety blocking for dangerous content, you can construct your
        LLM as follows:

        ```python
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            HarmBlockThreshold,
            HarmCategory,
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        ```

        For an enumeration of the categories and thresholds available, see Google's
        [safety settings](https://ai.google.dev/gemini-api/docs/safety-settings).

    ???+ example "Context caching"

        See [the docs](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#context-caching)
        for more info.

        Context caching allows you to store and reuse content (e.g., PDFs, images) for
        faster processing. The [`cached_content`][langchain_google_genai.ChatGoogleGenerativeAI.cached_content]
        parameter accepts a cache name created via the Google Generative AI API.

        See the Gemini docs for more details on [cached content](https://ai.google.dev/gemini-api/docs/caching?lang=python).

        Below are two examples: caching a single file directly and caching multiple
        files using `Part`.

        ???+ example "Single file example"

            This caches a single file and queries it.

            ```python
            from google import genai
            from google.genai import types
            import time
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.messages import HumanMessage

            client = genai.Client()

            # Upload file
            file = client.files.upload(file="path/to/your/file")
            while file.state.name == "PROCESSING":
                time.sleep(2)
                file = client.files.get(name=file.name)

            # Create cache
            model = "gemini-3.1-pro-preview"
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    display_name="Cached Content",
                    system_instruction=(
                        "You are an expert content analyzer, and your job is to answer "
                        "the user's query based on the file you have access to."
                    ),
                    contents=[file],
                    ttl="300s",
                ),
            )

            # Query with LangChain
            llm = ChatGoogleGenerativeAI(
                model=model,
                cached_content=cache.name,
            )
            message = HumanMessage(content="Summarize the main points of the content.")
            llm.invoke([message])
            ```

        ??? example "Multiple files example"

            This caches two files using `Part` and queries them together.

            ```python
            from google import genai
            from google.genai.types import CreateCachedContentConfig, Content, Part
            import time
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain.messages import HumanMessage

            client = genai.Client()

            # Upload files
            file_1 = client.files.upload(file="./file1")
            while file_1.state.name == "PROCESSING":
                time.sleep(2)
                file_1 = client.files.get(name=file_1.name)

            file_2 = client.files.upload(file="./file2")
            while file_2.state.name == "PROCESSING":
                time.sleep(2)
                file_2 = client.files.get(name=file_2.name)

            # Create cache with multiple files
            contents = [
                Content(
                    role="user",
                    parts=[
                        Part.from_uri(file_uri=file_1.uri, mime_type=file_1.mime_type),
                        Part.from_uri(file_uri=file_2.uri, mime_type=file_2.mime_type),
                    ],
                )
            ]
            model = "gemini-3.1-pro-preview"
            cache = client.caches.create(
                model=model,
                config=CreateCachedContentConfig(
                    display_name="Cached Contents",
                    system_instruction=(
                        "You are an expert content analyzer, and your job is to answer "
                        "the user's query based on the files you have access to."
                    ),
                    contents=contents,
                    ttl="300s",
                ),
            )

            # Query with LangChain
            llm = ChatGoogleGenerativeAI(
                model=model,
                cached_content=cache.name,
            )
            message = HumanMessage(
                content="Provide a summary of the key information across both files."
            )
            llm.invoke([message])
            ```

    ???+ example "Response metadata"

        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "model_name": "gemini-3.1-pro-preview",
            "model_provider": "google_genai",
            "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
            "finish_reason": "STOP",
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "probability": "NEGLIGIBLE",
                    "blocked": False,
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "probability": "NEGLIGIBLE",
                    "blocked": False,
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability": "NEGLIGIBLE",
                    "blocked": False,
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "NEGLIGIBLE",
                    "blocked": False,
                },
            ],
        }
        ```
    """  # noqa: E501

    client: Client | None = Field(
        default=None,
        exclude=True,  # Excluded from serialization
    )

    default_metadata: Sequence[tuple[str, str]] | None = Field(
        default=None,
        alias="default_metadata_input",
    )

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any unexpected initialization parameters."""

    streaming: bool | None = None
    """Whether to stream responses from the model."""

    convert_system_message_to_human: bool = False
    """Whether to merge any leading `SystemMessage` into the following `HumanMessage`.

    Gemini does not support system messages; any unsupported messages will raise an
    error.
    """

    _client_cleanup: _ClientCleanup = PrivateAttr()

    stop: list[str] | None = Field(default=None, alias="stop_sequences")
    """Stop sequences for the model."""

    response_mime_type: str | None = None
    """Output response MIME type of the generated candidate text.

    Supported MIME types:
        * `'text/plain'`: (default) Text output.
        * `'application/json'`: JSON response in the candidates.
        * `'text/x.enum'`: Enum in plain text. (legacy; use JSON schema output instead)

    !!! note

        The model also needs to be prompted to output the appropriate response type,
        otherwise the behavior is undefined.

        (In other words, simply setting this param doesn't force the model to comply;
        it only tells the model the kind of output expected. You still need to prompt it
        correctly.)
    """

    response_schema: dict[str, Any] | None = None
    """Enforce a schema to the output.

    The format of the dictionary should follow JSON Schema specification.

    !!! note "Schema Transformation"

        The Google GenAI SDK automatically transforms schemas for Gemini compatibility:

        - Inlines `$defs` definitions (enables Union types with `anyOf`)
        - Resolves `$ref` pointers for nested/recursive schemas
        - Preserves property ordering
        - Supports constraints like `minimum`/`maximum`, `minItems`/`maxItems`

    !!! tip "Using Union Types"

        Union types in Pydantic models (e.g., `field: Union[TypeA, TypeB]`) are
        automatically converted to `anyOf` schemas and work correctly with the
        `json_schema` method.

    Refer to the Gemini API [docs](https://ai.google.dev/gemini-api/docs/structured-output)
    for more details on supported JSON Schema features.
    """

    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = (
        Field(
            default=None,
            alias="thinking_level",
        )
    )
    """Indicates the thinking level.

    Possible values (support varies by model):
        * `'minimal'`: Lowest available reasoning depth.
        * `'low'`: Minimizes latency and cost.
        * `'medium'`: Balances latency/cost with reasoning depth.
        * `'high'`: Maximizes reasoning depth.
        * `'xhigh'`: Extra-high reasoning depth. Sent as-is only to models whose
            profile lists `'xhigh'` in `reasoning_effort_levels`; other models fall
            back to `'high'`.

    Check the model profile's `reasoning_effort_levels` and
    `reasoning_effort_default` fields for model-specific support. If those fields are
    unavailable, consult the upstream
    [Gemini API docs](https://ai.google.dev/gemini-api/docs/generate-content/thinking#thinking-levels-gemini-3).

    !!! note "Replaces `thinking_budget`"

        `thinking_budget` is deprecated for Gemini 3+ models. If both parameters are
        provided, this field takes precedence.

        If left unspecified, the model's default thinking level is used.

    !!! note "`thinking_level` alias"

        `thinking_level` -- Gemini's own native name for this setting -- is also
        accepted as an alias for this field, at both construction and call time.
        If both `thinking_level` and `reasoning_effort` are set, `thinking_level`
        wins (Pydantic's alias-resolution precedence). Use `reasoning_effort` or
        `thinking_level` interchangeably to read the value back.
    """

    thinking_config: dict[str, Any] | ThinkingConfig | None = Field(
        default=None,
    )
    """Raw Google GenAI thinking configuration.

    Accepts the same fields as `google.genai.types.ThinkingConfig`, including
    `thinking_level`, `thinking_budget`, and `include_thoughts`.

    !!! note "Precedence"

        If `thinking_config` is provided together with flat thinking arguments,
        the flat arguments take precedence for matching fields. After merging,
        `thinking_level` takes precedence over `thinking_budget` for Gemini 3+ models.
    """

    cached_content: str | None = None
    """The name of the cached content used as context to serve the prediction.

    !!! note

        Only used in explicit caching, where users can have control over caching (e.g.
        what content to cache) and enjoy guaranteed cost savings. Format:
        `cachedContents/{cachedContent}`.
    """

    use_interactions_api: bool | None = None
    """Whether to route requests through the Gemini Interactions API.

    If `None` (default), `ChatGoogleGenerativeAI` checks
    `LANGCHAIN_GOOGLE_USE_INTERACTIONS_API`, `output_version="interactions/v1"`, and
    auto-routes when Interactions-specific options (`use_previous_interaction_id`,
    `previous_interaction_id`, `continuation_token`, `automatic_continuation`, `agent`,
    `agent_config`, `background`, `environment`, `webhook_config`, `deep-research-*` /
    `antigravity-*` models, or `file_search` / `mcp_server` / `retrieval` server tools)
    are active.

    !!! warning "Beta feature"

        The Gemini Interactions API is currently in public beta (`v1beta` / `v1beta1`)
        and its wire schema and preview capabilities may evolve across releases.
    """

    use_previous_interaction_id: bool = False
    """Whether to chain stateful multi-turn sessions via `previous_interaction_id`.

    When `True`, enables server-side interaction storage (`store=True`) unless
    explicitly overridden and sends only messages following the most recent stored
    Google `AIMessage` alongside `previous_interaction_id`.

    !!! warning "Beta feature"

        Stateful session chaining relies on the Gemini Interactions API (`store=True`).
    """

    store: bool | None = None
    """Whether the Interactions API should persist the interaction server-side.

    When `None`, defaults to `True` if stateful chaining, `background=True`, or `agent`
    execution is active, and `False` otherwise for zero data retention parity with
    `generateContent`.
    """

    service_tier: Literal["flex", "standard", "priority"] | ServiceTier | str | None = (
        None
    )
    """Request pricing and performance tier (`'flex'`, `'standard'`, or `'priority'`).

    Applies to both `generateContent` (sent as `service_tier` on the Gemini Developer
    API and as the `X-Vertex-AI-LLM-Shared-Request-Type` header on Vertex AI) and the
    Interactions API. Can be overridden per request via invocation kwargs.
    """

    agent: str | None = None
    """Name of a server-hosted Gemini agent (e.g. `'deep-research-preview-04-2026'`).

    When specified, creates an agent interaction (`CreateAgentInteraction`) instead of a
    standard model interaction.

    !!! warning "Experimental"

        Agent interactions are in preview in `google-genai` and may change in future
        versions.
    """

    agent_config: dict[str, Any] | None = None
    """Configuration dictionary passed to `CreateAgentInteraction`.

    !!! warning "Experimental"

        Agent configuration schemas are in preview and may change in future versions.
    """

    environment: str | dict[str, Any] | None = None
    """Execution environment identifier or configuration for Interactions runs.

    !!! warning "Experimental"

        Interactions execution environments are in preview and may change in future
        versions.
    """

    background: bool | None = None
    """Whether to run the interaction asynchronously in the background."""

    polling_interval: float | None = None
    """Seconds between `interactions.get` polls for background runs (`2.0` default)."""

    def __init__(self, **kwargs: Any) -> None:
        """Needed for arg validation."""
        # Get all valid field names, including aliases
        valid_fields = set()
        for field_name, field_info in self.__class__.model_fields.items():
            valid_fields.add(field_name)
            if hasattr(field_info, "alias") and field_info.alias is not None:
                valid_fields.add(field_info.alias)

        # Check for unrecognized arguments
        for arg in kwargs:
            if arg not in valid_fields:
                suggestions = get_close_matches(arg, valid_fields, n=1)
                suggestion = (
                    f" Did you mean: '{suggestions[0]}'?" if suggestions else ""
                )
                logger.warning(
                    f"Unexpected argument '{arg}' "
                    f"provided to ChatGoogleGenerativeAI.{suggestion}"
                )
        super().__init__(**kwargs)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai"

    @property
    def thinking_level(
        self,
    ) -> Literal["minimal", "low", "medium", "high", "xhigh"] | None:
        """Alias for `reasoning_effort` (Gemini's native name for this setting)."""
        return self.reasoning_effort

    @property
    def _supports_xhigh_thinking_level(self) -> bool:
        levels = (self.profile or {}).get("reasoning_effort_levels") or []
        return "xhigh" in levels

    @property
    def _supports_code_execution(self) -> bool:
        """Whether the model supports code execution.

        See [Gemini models](https://ai.google.dev/gemini-api/docs/models) for a list.
        """
        # TODO: Refactor to use `capabilities` property when supported upstream
        # (or done via augmentation)
        model = _normalize_gemini_model_name(self.model)
        return model.startswith(
            (
                "gemini-2-",
                "gemini-2.",
                "gemini-3-",
                "gemini-3.",
                "gemini-4-",
                "gemini-4.",
            )
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @field_validator("thinking_config", mode="before")
    @classmethod
    def _serialize_thinking_config(
        cls, value: ThinkingConfig | dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if isinstance(value, ThinkingConfig):
            return value.model_dump(exclude_none=True)
        return value

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in.

        (In other words, handle additional params that aren't explicitly defined as
        model fields. Used to pass extra config to underlying APIs without defining them
        all here.)
        """
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def _set_langchain_google_genai_version(self) -> Self:
        """Set package version in metadata."""
        self._add_version("langchain-google-genai", __version__)
        return self

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and builds client."""
        if "temperature" not in self.model_fields_set and _is_gemini_3_or_later(
            self.model
        ):
            self.temperature = None

        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            msg = "temperature must be in the range [0.0, 2.0]"
            raise ValueError(msg)

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            msg = "top_p must be in the range [0.0, 1.0]"
            raise ValueError(msg)

        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        if (
            self.frequency_penalty is not None
            and not -2.0 <= self.frequency_penalty <= 2.0
        ):
            msg = "frequency_penalty must be in the range [-2.0, 2.0]"
            raise ValueError(msg)

        if (
            self.presence_penalty is not None
            and not -2.0 <= self.presence_penalty <= 2.0
        ):
            msg = "presence_penalty must be in the range [-2.0, 2.0]"
            raise ValueError(msg)

        if self.polling_interval is not None and self.polling_interval <= 0:
            msg = "polling_interval must be positive"
            raise ValueError(msg)

        additional_headers = self.additional_headers or {}
        self.default_metadata = tuple(additional_headers.items())

        _, user_agent = get_user_agent("ChatGoogleGenerativeAI")
        headers = {"user-agent": user_agent, **additional_headers}

        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key

        base_url = self.base_url
        if isinstance(self.base_url, dict):
            # Handle case where base_url is provided as a dict
            # (Backwards compatibility for deprecated client_options field)
            if keys := list(self.base_url.keys()):
                if "api_endpoint" in keys and len(keys) == 1:
                    base_url = self.base_url["api_endpoint"]
                elif "api_endpoint" in keys and len(keys) > 1:
                    msg = (
                        "When providing base_url as a dict, it can only contain the "
                        "api_endpoint key. Extra keys found: "
                        f"{[k for k in keys if k != 'api_endpoint']}"
                    )
                    raise ValueError(msg)
                else:
                    msg = (
                        "When providing base_url as a dict, it must only contain the "
                        "api_endpoint key."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    "base_url must be a string or a dict containing the "
                    "api_endpoint key."
                )
                raise ValueError(msg)

        http_options = HttpOptions(
            base_url=cast("str", base_url),
            api_version=_default_api_version(
                self.api_version,
                vertexai=self._use_vertexai,  # type: ignore[attr-defined]
            ),
            headers=headers,
            client_args=self.client_args,
            async_client_args=self.client_args,
        )

        if self._use_vertexai:  # type: ignore[attr-defined]
            # Vertex AI backend - supports both API key and credentials
            # Note: The google-genai SDK requires API keys to be passed via environment
            # variables when using Vertex AI, not via the api_key parameter.
            # If an API key is provided programmatically, we set it in the environment
            # temporarily for the Client initialization.

            # Normalize model name for Vertex AI - strip 'models/' prefix
            # Vertex AI expects model names without the prefix
            # (e.g., "gemini-3.5-flash") while Google AI accepts both formats
            if self.model.startswith("models/"):
                object.__setattr__(self, "model", self.model.replace("models/", "", 1))

            api_key_env_set = False

            if (
                google_api_key
                and not os.getenv("GOOGLE_API_KEY")
                and not os.getenv("GEMINI_API_KEY")
            ):
                # Set the API key in environment for Client initialization
                os.environ["GOOGLE_API_KEY"] = google_api_key
                api_key_env_set = True

            try:
                self.client = Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                    credentials=self.credentials,
                    http_options=http_options,
                )
            finally:
                # Clean up the temporary environment variable if we set it
                if api_key_env_set:
                    os.environ.pop("GOOGLE_API_KEY", None)
        else:
            # Gemini Developer API - requires API key
            if not google_api_key:
                msg = (
                    "API key required for Gemini Developer API. Provide api_key "
                    "parameter or set GOOGLE_API_KEY/GEMINI_API_KEY environment "
                    "variable."
                )
                raise ValueError(msg)
            self.client = Client(api_key=google_api_key, http_options=http_options)
        self._client_cleanup = _ClientCleanup(self.client)
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            model_id = re.sub(r"-\d{3}$", "", _normalize_gemini_model_name(self.model))
            self.profile = _get_default_model_profile(model_id)
        return self

    def _ensure_client_cleanup(self) -> _ClientCleanup | None:
        cleanup = getattr(self, "_client_cleanup", None)
        if isinstance(cleanup, _ClientCleanup):
            return cleanup
        if isinstance(self.client, Client):
            with _FALLBACK_INTERACTIONS_LOCK:
                cleanup = getattr(self, "_client_cleanup", None)
                if not isinstance(cleanup, _ClientCleanup):
                    cleanup = _ClientCleanup(self.client)
                    object.__setattr__(self, "_client_cleanup", cleanup)
            return cleanup
        return None

    @property
    def async_client(self) -> Any:
        """Async client for Google GenAI operations..

        Returns:
            The async client interface that exposes async versions of all client
                methods.

        Raises:
            ValueError: If the client has not been initialized.
        """
        if self.client is None:
            msg = "Client not initialized. Initialize the model first."
            raise ValueError(msg)
        cleanup = self._ensure_client_cleanup()
        if cleanup is not None:
            cleanup.register_async_loop()
        return self.client.aio

    async def aclose(self) -> None:
        """Close the sync and async clients on the async client's event loop.

        Call this method before the event loop used for async requests exits.

        Raises:
            RuntimeError: If called from a different event loop than the one used for
                async requests.
        """
        cleanup = self._ensure_client_cleanup()
        if cleanup is not None:
            await cleanup.aclose()

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "n": self.n,
            "safety_settings": self.safety_settings,
            "response_modalities": self.response_modalities,
            "media_resolution": self.media_resolution,
            "thinking_budget": self.thinking_budget,
            "include_thoughts": self.include_thoughts,
            "thinking_level": self.reasoning_effort,
            "thinking_config": self.thinking_config,
            "image_config": self.image_config,
            "use_interactions_api": self.use_interactions_api,
            "use_previous_interaction_id": self.use_previous_interaction_id,
            "store": self.store,
            "service_tier": self.service_tier,
            "agent": self.agent,
            "agent_config": self.agent_config,
            "environment": self.environment,
            "background": self.background,
        }

    def _use_interactions_api(self, kwargs: Mapping[str, Any]) -> bool:
        """Determine whether to route a request through the Gemini Interactions API.

        Args:
            kwargs: Per-call invocation keyword arguments.

        Returns:
            `True` when the request should use `client.interactions`, `False` when it
            should use `client.models.generate_content`.
        """
        kw_flag = kwargs.get("use_interactions_api")
        if isinstance(kw_flag, bool):
            return kw_flag
        mk_flag = self.model_kwargs.get("use_interactions_api")
        if isinstance(mk_flag, bool):
            return mk_flag
        if isinstance(self.use_interactions_api, bool):
            return self.use_interactions_api

        env_val = os.environ.get("LANGCHAIN_GOOGLE_USE_INTERACTIONS_API")
        if env_val is not None:
            normalized_env = env_val.strip().lower()
            if normalized_env in {"1", "true", "yes"}:
                return True
            if normalized_env in {"0", "false", "no"}:
                return False

        if (
            self.output_version == "interactions/v1"
            or kwargs.get("output_version") == "interactions/v1"
        ):
            return True

        if (
            self.use_previous_interaction_id
            or bool(kwargs.get("use_previous_interaction_id"))
            or bool(self.model_kwargs.get("use_previous_interaction_id"))
        ):
            return True

        if (
            kwargs.get("previous_interaction_id") is not None
            or self.model_kwargs.get("previous_interaction_id") is not None
            or kwargs.get("continuation_token") is not None
            or self.model_kwargs.get("continuation_token") is not None
            or kwargs.get("automatic_continuation") is not None
            or self.model_kwargs.get("automatic_continuation") is not None
        ):
            return True

        for attr_name in (
            "agent",
            "agent_config",
            "background",
            "environment",
        ):
            if (
                getattr(self, attr_name) is not None
                or kwargs.get(attr_name) is not None
                or self.model_kwargs.get(attr_name) is not None
            ):
                return True

        if (
            kwargs.get("webhook_config") is not None
            or self.model_kwargs.get("webhook_config") is not None
        ):
            return True

        effective_model = (
            kwargs.get("model") or self.model_kwargs.get("model") or self.model
        )
        if isinstance(effective_model, str):
            stripped_model = _normalize_gemini_model_name(effective_model)
            if stripped_model.startswith(_INTERACTIONS_AUTO_MODEL_PREFIXES):
                return True

        for tool_source in (kwargs.get("tools"), self.model_kwargs.get("tools")):
            if isinstance(tool_source, Sequence) and not isinstance(
                tool_source, (str, bytes)
            ):
                for tool_item in tool_source:
                    if (
                        isinstance(tool_item, Mapping)
                        and tool_item.get("type") in _INTERACTIONS_ONLY_TOOL_TYPES
                    ):
                        return True

        return False

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        code_execution: bool | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Override `invoke` on `ChatGoogleGenerativeAI` to add `code_execution`."""
        if code_execution is not None:
            if not self._supports_code_execution:
                msg = (
                    "Code execution is only supported on Gemini 2, 3, and 4 models. "
                    f"Current model: {self.model}"
                )
                raise ValueError(msg)
            if "tools" not in kwargs:
                code_execution_tool = GoogleTool(code_execution=ToolCodeExecution())
                kwargs["tools"] = [code_execution_tool]

            else:
                msg = "Tools are already defined.code_execution tool can't be defined"
                raise ValueError(msg)

        return super().invoke(input, config, stop=stop, **kwargs)

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Stream response chunks asynchronously, closing inner streams cleanly."""
        active_streams: list[Any] = []
        outer_stream = super().astream(input, config=config, stop=stop, **kwargs)
        try:
            async with _safe_aclosing(outer_stream) as managed_outer:
                aiterator = managed_outer.__aiter__()
                while True:
                    token = _ACTIVE_ASTREAM_SINK.set(active_streams)
                    try:
                        chunk = await aiterator.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        _ACTIVE_ASTREAM_SINK.reset(token)
                    yield chunk
        finally:
            for inner_stream in reversed(active_streams):
                await _aclose_async_iterator(inner_stream)

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get invocation parameters with sensitive MCP headers redacted."""
        params = super()._get_invocation_params(stop=stop, **kwargs)
        params.pop("_structured_output_schema", None)
        if "tools" in params:
            params["tools"] = _redact_mcp_tools(params["tools"])
        model_kw = params.get("model_kwargs")
        if isinstance(model_kw, Mapping) and "tools" in model_kw:
            redacted_tools = _redact_mcp_tools(model_kw["tools"])
            if redacted_tools is not model_kw["tools"]:
                copied_mk = dict(model_kw)
                copied_mk["tools"] = redacted_tools
                params["model_kwargs"] = copied_mk
        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        models_prefix = "models/"
        raw_model: str | None
        if self._use_interactions_api(kwargs) and (
            active_agent := kwargs.get("agent")
            or self.agent
            or self.model_kwargs.get("agent")
        ):
            raw_model = str(active_agent)
        else:
            raw_model = params.get("model") or self.model
        ls_model_name = (
            raw_model[len(models_prefix) :]
            if raw_model and raw_model.startswith(models_prefix)
            else raw_model
        )
        ls_params = LangSmithParams(
            ls_provider="google_genai",
            ls_model_name=ls_model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _supports_thinking(self) -> bool:
        """Check if the current model supports thinking capabilities."""
        return self.profile.get("reasoning_output", False) if self.profile else False

    def _prepare_params(
        self,
        stop: list[str] | None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationConfig:
        """Prepare generation parameters with config logic."""
        gen_config = self._build_base_generation_config(stop, **kwargs)
        if generation_config:
            gen_config = self._merge_generation_config(gen_config, generation_config)

        # Handle response-specific kwargs (MIME type and structured output)
        gen_config = self._add_response_parameters(gen_config, **kwargs)

        return GenerationConfig.model_validate(gen_config)

    def _build_base_generation_config(
        self, stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        """Build the base generation configuration from instance attributes."""
        config: dict[str, Any] = {
            "candidate_count": kwargs.get("candidate_count", self.n),
            "temperature": kwargs.get("temperature", self.temperature),
            "stop_sequences": (
                stop if stop is not None else kwargs.get("stop_sequences", self.stop)
            ),
            "max_output_tokens": kwargs.get(
                "max_output_tokens", self.max_output_tokens
            ),
            "top_k": kwargs.get("top_k", self.top_k),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.frequency_penalty
            ),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "response_modalities": kwargs.get(
                "response_modalities", self.response_modalities
            ),
            "seed": kwargs.get("seed", self.seed),
        }

        # Convert response modalities
        config["response_modalities"] = (
            [m.value for m in config["response_modalities"]]
            if config["response_modalities"]
            else None
        )

        # Auto-set audio output and speech_config for TTS models
        # if not explicitly configured
        if self.model.endswith("-tts") or "-tts-" in self.model:
            if config["response_modalities"] is None:
                config["response_modalities"] = ["AUDIO"]
            if config.get("speech_config") is None:
                voice_name = config.pop("voice_name", "Kore")
                config["speech_config"] = SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )

        thinking_config = self._build_thinking_config(**kwargs)
        if thinking_config is not None:
            config["thinking_config"] = thinking_config

        return {k: v for k, v in config.items() if v is not None}

    def _build_thinking_config(self, **kwargs: Any) -> ThinkingConfig | None:
        """Build thinking configuration if supported by the model."""
        raw_thinking_config = kwargs.get("thinking_config", self.thinking_config)
        # `thinking_level` is Gemini's native alias for `reasoning_effort`
        # (`Field(alias="thinking_level")`). Pydantic aliases don't apply to
        # call-time kwargs, so it's resolved here too. `thinking_level` wins if
        # both are set, matching the construction-time alias-resolution
        # precedence.
        thinking_level = kwargs.get(
            "thinking_level", kwargs.get("reasoning_effort", self.reasoning_effort)
        )
        if (
            isinstance(thinking_level, str)
            and thinking_level.lower() == "xhigh"
            and not self._supports_xhigh_thinking_level
        ):
            # Models without `xhigh` reject it, so fall back to the closest level.
            warnings.warn(
                f"Model '{self.model}' does not list 'xhigh' in its profile's "
                "reasoning_effort_levels; using 'high' instead. Set a profile with "
                "'xhigh' in reasoning_effort_levels to send it as-is.",
                UserWarning,
                stacklevel=2,
            )
            thinking_level = "high"
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        include_thoughts = kwargs.get("include_thoughts", self.include_thoughts)

        has_thinking_params = (
            raw_thinking_config is not None
            or thinking_level is not None
            or thinking_budget is not None
            or include_thoughts is not None
        )
        if not has_thinking_params:
            return None

        config: dict[str, Any] = {}
        if raw_thinking_config is not None:
            config.update(
                ThinkingConfig.model_validate(raw_thinking_config).model_dump(
                    exclude_none=True
                )
            )

        if thinking_level is not None:
            config["thinking_level"] = thinking_level
        if thinking_budget is not None:
            config["thinking_budget"] = thinking_budget
        if include_thoughts is not None:
            config["include_thoughts"] = include_thoughts

        # thinking_level takes precedence over thinking_budget for Gemini 3+ models
        if "thinking_level" in config and "thinking_budget" in config:
            warnings.warn(
                "Both 'thinking_level' and 'thinking_budget' were set after merging "
                "thinking configuration values. 'thinking_level' takes precedence "
                "for Gemini 3+ models; 'thinking_budget' will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            config.pop("thinking_budget")

        return ThinkingConfig(**config)

    def _merge_generation_config(
        self, base_config: dict[str, Any], generation_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user-provided generation config with base config."""
        processed_config = dict(generation_config)
        # Convert string response_modalities to Modality enums if needed
        if "response_modalities" in processed_config:
            modalities = processed_config["response_modalities"]
            if (
                isinstance(modalities, list)
                and modalities
                and isinstance(modalities[0], str)
            ):
                from langchain_google_genai import Modality

                try:
                    processed_config["response_modalities"] = [
                        getattr(Modality, modality) for modality in modalities
                    ]
                except AttributeError as e:
                    msg = f"Invalid response modality: {e}"
                    raise ValueError(msg) from e
        return {**base_config, **processed_config}

    def _add_response_parameters(
        self, gen_config: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Add response-specific parameters to generation config.

        Includes `response_mime_type`, `response_schema`, and `response_json_schema`.
        """
        # Handle response mime type
        response_mime_type = kwargs.get("response_mime_type", self.response_mime_type)
        if response_mime_type is not None:
            gen_config["response_mime_type"] = response_mime_type

        response_schema = kwargs.get("response_schema", self.response_schema)
        response_json_schema = kwargs.get("response_json_schema")  # If passed as kwarg

        # Handle both response_schema and response_json_schema
        # (Regardless, we use `response_json_schema` in the request)
        schema_to_use = (
            response_json_schema
            if response_json_schema is not None
            else response_schema
        )
        if schema_to_use:
            self._validate_and_add_response_schema(
                gen_config=gen_config,
                response_schema=schema_to_use,
                response_mime_type=response_mime_type,
            )

        return gen_config

    def _validate_and_add_response_schema(
        self,
        gen_config: dict[str, Any],
        response_schema: dict[str, Any],
        response_mime_type: str | None,
    ) -> None:
        """Validate and add response schema to generation config."""
        if response_mime_type != "application/json":
            error_message = (
                "JSON schema structured output is only supported when "
                "response_mime_type is set to 'application/json'"
            )
            if response_mime_type == "text/x.enum":
                error_message += (
                    ". Instead of 'text/x.enum', define enums using your JSON schema."
                )
            raise ValueError(error_message)

        gen_config["response_json_schema"] = response_schema

    def _prepare_request(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare the request configuration for the API call."""
        # Process tools and functions
        formatted_tools = self._format_tools(tools, functions)

        # Remove any messages with empty content
        filtered_messages = self._filter_messages(messages)

        # Parse chat history into Gemini Content
        system_instruction, history = _parse_chat_history(
            filtered_messages,
            convert_system_message_to_human=self.convert_system_message_to_human,
            model=self.model,
            use_vertexai=self._use_vertexai,  # type: ignore[attr-defined]
        )
        if (
            _uses_fixed_sampling_and_disallows_prefill(self.model)
            and history
            and history[-1].role == "model"
        ):
            msg = (
                f"Model '{self.model}' does not support model prefilling. The final "
                "request turn must be a user message or a function response."
            )
            raise ValueError(msg)

        # Bridge inline latitude/longitude from Interactions-shaped google_maps tool
        effective_tool_config = tool_config
        if effective_tool_config is None and tools:
            for raw_t in tools:
                if (
                    isinstance(raw_t, Mapping)
                    and raw_t.get("type") == "google_maps"
                    and raw_t.get("latitude") is not None
                    and raw_t.get("longitude") is not None
                ):
                    raw_map_t = cast("Mapping[str, Any]", raw_t)
                    effective_tool_config = {
                        "retrieval_config": {
                            "lat_lng": {
                                "latitude": float(raw_map_t["latitude"]),
                                "longitude": float(raw_map_t["longitude"]),
                            }
                        }
                    }
                    break

        # Process tool configuration
        formatted_tool_config = self._process_tool_config(
            tool_choice, effective_tool_config, formatted_tools
        )

        # Process safety settings
        formatted_safety_settings = self._format_safety_settings(
            safety_settings if safety_settings is not None else self.safety_settings
        )

        timeout = kwargs.pop("timeout", None)
        if timeout is not None:
            timeout = int(timeout * 1000)
        elif self.timeout is not None:
            timeout = int(self.timeout * 1000)

        max_retries = kwargs.pop("max_retries", None)
        if max_retries is None:
            max_retries = self.max_retries

        # Handle OpenAI-style `strict` kwarg (used by langchain.agents.create_agent)
        # Google's json_schema is inherently strict, so we just consume this.
        kwargs.pop("strict", None)
        kwargs.pop("use_interactions_api", None)

        # Handle OpenAI-style `response_format` kwarg
        # Ref: https://platform.openai.com/docs/guides/structured-outputs
        # Compatible with langchain.agents.create_agent ProviderStrategy
        response_format = kwargs.pop("response_format", None)
        if response_format is not None and isinstance(response_format, dict):
            rf_type = response_format.get("type")
            if rf_type in ("json_object", "json_schema"):
                if "response_mime_type" not in kwargs:
                    kwargs["response_mime_type"] = "application/json"
                json_schema = response_format.get("json_schema", {})
                schema = json_schema.get("schema")
                if schema and "response_json_schema" not in kwargs:
                    kwargs["response_json_schema"] = schema

        # Get generation parameters
        # (consumes thinking kwargs into params.thinking_config)
        params: GenerationConfig = self._prepare_params(
            stop, generation_config=generation_config, **kwargs
        )

        image_config = kwargs.pop("image_config", None)

        labels = kwargs.pop("labels", None)
        if labels is None:
            labels = self.labels

        service_tier = kwargs.pop("service_tier", None)
        if service_tier is None:
            service_tier = self.service_tier
        if service_tier is None:
            service_tier = self.model_kwargs.get("service_tier")

        _consumed_kwargs = {
            "thinking_budget",
            "thinking_level",
            "include_thoughts",
            "reasoning_effort",
            "response_schema",
            "response_json_schema",
            "response_mime_type",
            "_structured_output_schema",
        }
        _consumed_kwargs.update(params.model_fields_set)
        # Filter out kwargs already consumed by _prepare_params.
        # These are handled via params and aren't direct fields
        # on GenerateContentConfig or would cause duplicate argument errors.
        remaining_kwargs = {
            k: v for k, v in kwargs.items() if k not in _consumed_kwargs
        }

        # Build request configuration
        request = self._build_request_config(
            formatted_tools,
            formatted_tool_config,
            formatted_safety_settings,
            params,
            cached_content,
            system_instruction,
            timeout=timeout,
            max_retries=max_retries,
            image_config=image_config,
            labels=labels,
            service_tier=service_tier,
            **remaining_kwargs,
        )

        # Return config and additional params needed for API call
        return {"model": self.model, "contents": history, "config": request}

    def _format_tools(
        self,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
    ) -> list | None:
        """Format tools and functions for the API."""
        code_execution_tool = GoogleTool(code_execution=ToolCodeExecution())
        if tools == [code_execution_tool]:
            return list(tools)
        if tools:
            return convert_to_genai_function_declarations(tools)
        if functions:
            return convert_to_genai_function_declarations(functions)
        return None

    def _filter_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Normalize messages with empty content."""
        filtered_messages: list[BaseMessage] = []
        for message in messages:
            if isinstance(message, HumanMessage) and not message.content:
                warnings.warn(
                    "HumanMessage with empty content was removed to prevent API error"
                )
            elif (
                self._use_vertexai  # type: ignore[attr-defined]
                and isinstance(message, AIMessage)
                and message.content == []
            ):
                filtered_messages.append(
                    message.model_copy(
                        update={"content": [{"type": "text", "text": ""}]}
                    )
                )
            else:
                filtered_messages.append(message)
        return filtered_messages

    def _process_tool_config(
        self,
        tool_choice: _ToolChoiceType | bool | None,
        tool_config: dict | ToolConfig | None,
        formatted_tools: list | None,
    ) -> ToolConfig | None:
        """Process tool configuration and choice.

        Merges `tool_choice` and `tool_config` when both are provided.

        `tool_choice` controls `function_calling_config` while `tool_config` can provide
        retrieval_config (for Maps/Search grounding) and other configurations.
        """
        # Normalize tool_config to ToolConfig object if dict
        normalized_config: ToolConfig | None = None
        if tool_config:
            normalized_config = (
                ToolConfig.model_validate(tool_config)
                if isinstance(tool_config, dict)
                else tool_config
            )

        # Check for conflicts
        if tool_choice and normalized_config:
            if normalized_config.function_calling_config:
                msg = (
                    "Cannot specify both tool_choice and "
                    "tool_config.function_calling_config. "
                    f"Received {tool_choice=} and "
                    f"tool_config.function_calling_config="
                    f"{normalized_config.function_calling_config}"
                )
                raise ValueError(msg)

        # Process tool_choice
        if tool_choice:
            if not formatted_tools:
                msg = (
                    f"Received {tool_choice=} but no {formatted_tools=}. "
                    "'tool_choice' can only be specified if 'tools' is specified."
                )
                raise ValueError(msg)
            all_names = self._extract_tool_names(formatted_tools)

            # Only set function_calling_config if there are actual callable functions
            # (built-in tools like google_maps don't have function_declarations)
            if not all_names:
                # No callable functions, only built-in tools like Maps/Search
                # Just pass through tool_config without function_calling_config
                if normalized_config:
                    return normalized_config
                return None

            choice_config = _tool_choice_to_tool_config(tool_choice, all_names)

            # Merge with tool_config if it exists
            if normalized_config:
                # Merge: take function_calling_config from choice_config
                # and other fields from normalized_config
                return ToolConfig(
                    function_calling_config=choice_config.function_calling_config,
                    retrieval_config=normalized_config.retrieval_config,
                )
            return choice_config

        # Only tool_config provided
        if normalized_config:
            return normalized_config

        return None

    def _extract_tool_names(self, formatted_tools: list) -> list[str]:
        """Extract tool names from formatted tools."""
        all_names: list[str] = []
        for t in formatted_tools:
            if hasattr(t, "function_declarations") and t.function_declarations:
                t_with_declarations = cast("Any", t)
                all_names.extend(
                    f.name for f in t_with_declarations.function_declarations
                )
            elif isinstance(t, GoogleTool):
                # Built-in tools like code_execution, google_search, google_maps
                # don't have function_declarations
                continue
            else:
                msg = f"Tool {t} doesn't have function_declarations attribute"
                raise TypeError(msg)
        return all_names

    def _format_safety_settings(
        self, safety_settings: SafetySettingDict | None
    ) -> list[SafetySetting]:
        """Format safety settings for the API."""
        if not safety_settings:
            return []
        if isinstance(safety_settings, dict):
            return [
                SafetySetting(category=category, threshold=threshold)
                for category, threshold in safety_settings.items()
            ]
        msg = "safety_settings must be: dict[HarmCategory, HarmBlockThreshold]"
        raise TypeError(msg)

    def _build_request_config(
        self,
        formatted_tools: list | None,
        formatted_tool_config: ToolConfig | None,
        formatted_safety_settings: list[SafetySetting],
        params: GenerationConfig,
        cached_content: str | None,
        system_instruction: Content | None,
        timeout: int | None = None,
        max_retries: int | None = None,
        image_config: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
        service_tier: ServiceTier | str | None = None,
        **kwargs: Any,
    ) -> GenerateContentConfig:
        """Build the final request configuration."""

        # Pop any per-request `http_options` so it can be merged with the
        # internally-derived timeout/retry options below. Passing it through to
        # `GenerateContentConfig` directly would collide with the explicit
        # `http_options` keyword argument and raise a duplicate-keyword error.
        per_request_http_options = kwargs.pop("http_options", None)
        if isinstance(per_request_http_options, dict):
            per_request_http_options = HttpOptions(**per_request_http_options)

        retry_options = None
        if max_retries is not None:
            retry_options = HttpRetryOptions(attempts=max_retries)

        http_options = None
        if timeout is not None or retry_options is not None:
            http_options = HttpOptions(
                timeout=timeout,
                retry_options=retry_options,
            )

        # Vertex AI `generateContent` selects the billing tier via header rather than
        # the body `service_tier` field. Merge before `per_request_http_options` so
        # explicit caller-supplied headers still take precedence.
        config_service_tier: Any = service_tier
        if service_tier is not None and self._use_vertexai:  # type: ignore[attr-defined]
            config_service_tier = None
            tier_header_val = (
                str(service_tier.value)
                if isinstance(service_tier, Enum)
                else str(service_tier)
            ).lower()
            http_options = _merge_http_options(
                http_options,
                HttpOptions(headers={_VERTEX_SERVICE_TIER_HEADER: tier_header_val}),
            )

        if per_request_http_options is not None:
            http_options = _merge_http_options(http_options, per_request_http_options)

        image_config_dict = (
            image_config if image_config is not None else self.image_config
        )
        image_config_obj = None
        if image_config_dict is not None:
            image_config_obj = ImageConfig(**image_config_dict)

        request_params = params.model_dump(exclude_unset=True)
        if _uses_fixed_sampling_and_disallows_prefill(self.model):
            # These models use fixed sampling defaults and ignore custom sampling
            # parameters. Drop them from the request and warn so an explicitly-set
            # value isn't silently discarded (consistent with how superseded
            # thinking parameters are handled above).
            ignored_parameters = [
                parameter
                for parameter in _CUSTOM_SAMPLING_PARAMETERS
                if parameter in request_params
            ]
            if ignored_parameters:
                warnings.warn(
                    f"Model '{self.model}' uses fixed sampling defaults; the "
                    f"sampling parameter(s) {', '.join(ignored_parameters)} "
                    "will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                for parameter in ignored_parameters:
                    request_params.pop(parameter, None)

        return GenerateContentConfig(
            tools=list(formatted_tools) if formatted_tools else None,
            tool_config=formatted_tool_config,
            safety_settings=formatted_safety_settings,
            cached_content=cached_content,
            system_instruction=system_instruction,
            http_options=http_options,
            image_config=image_config_obj,
            labels=labels,
            service_tier=config_service_tier,
            **request_params,
            **kwargs,
        )

    def _generate_interactions(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute a synchronous request via the Gemini Interactions API."""
        from langchain_google_genai._interactions_response import (
            generate_interactions,
        )

        return generate_interactions(
            self,
            messages,
            stop=stop,
            run_manager=run_manager,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def _agenerate_interactions(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Execute an asynchronous request via the Gemini Interactions API."""
        from langchain_google_genai._interactions_response import (
            agenerate_interactions,
        )

        return await agenerate_interactions(
            self,
            messages,
            stop=stop,
            run_manager=run_manager,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )

    def _stream_interactions(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chunks synchronously from the Gemini Interactions API."""
        msg = "Gemini Interactions API streaming is not yet wired."
        raise NotImplementedError(msg)

    async def _astream_interactions(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chunks asynchronously from the Gemini Interactions API."""
        msg = "Gemini Interactions API async streaming is not yet wired."
        raise NotImplementedError(msg)
        yield  # pragma: no cover

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)

        routing_kwargs = {**kwargs, "tools": tools} if tools is not None else kwargs
        use_interactions = self._use_interactions_api(routing_kwargs)
        kwargs.pop("use_interactions_api", None)
        if use_interactions:
            return self._generate_interactions(
                messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        try:
            response: GenerateContentResponse = self.client.models.generate_content(
                **request,
            )
        except ClientError as e:
            _handle_client_error(e, request)
        except ServerError as e:
            _handle_server_error(e)

        return _response_to_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)

        routing_kwargs = {**kwargs, "tools": tools} if tools is not None else kwargs
        use_interactions = self._use_interactions_api(routing_kwargs)
        kwargs.pop("use_interactions_api", None)
        if use_interactions:
            return await self._agenerate_interactions(
                messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        try:
            response: GenerateContentResponse = (
                await self.async_client.models.generate_content(
                    **request,
                )
            )
        except ClientError as e:
            _handle_client_error(e, request)
        except ServerError as e:
            _handle_server_error(e)

        return _response_to_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)

        routing_kwargs = {**kwargs, "tools": tools} if tools is not None else kwargs
        use_interactions = self._use_interactions_api(routing_kwargs)
        kwargs.pop("use_interactions_api", None)
        if use_interactions:
            yield from self._stream_interactions(
                messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            return

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        response: Iterator[GenerateContentResponse] = (
            self.client.models.generate_content_stream(
                **request,
            )
        )

        prev_usage_metadata: UsageMetadata | None = None  # Cumulative usage
        indexer = _StreamBlockIndexer()
        saw_tool_call = False
        for chunk in _classified_stream(response, request):
            if chunk:
                _chat_result = _response_to_result(
                    chunk, stream=True, prev_usage=prev_usage_metadata
                )
                gen = cast("ChatGenerationChunk", _chat_result.generations[0])
                message = cast("AIMessageChunk", gen.message)

            # Populate index if missing
            indexer.assign(message)
            saw_tool_call = _mark_streamed_tool_call(gen, saw_tool_call)

            prev_usage_metadata = (
                message.usage_metadata
                if prev_usage_metadata is None
                else add_usage(prev_usage_metadata, message.usage_metadata)
            )

            if run_manager:
                run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | ToolConfig | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)

        routing_kwargs = {**kwargs, "tools": tools} if tools is not None else kwargs
        use_interactions = self._use_interactions_api(routing_kwargs)
        kwargs.pop("use_interactions_api", None)
        if use_interactions:
            ia_stream = self._astream_interactions(
                messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
                cached_content=cached_content,
                tool_choice=tool_choice,
                **kwargs,
            )
            if (active_sink := _ACTIVE_ASTREAM_SINK.get()) is not None:
                active_sink.append(ia_stream)
            async with _safe_aclosing(ia_stream) as managed_ia_stream:
                async for ichunk in managed_ia_stream:
                    yield ichunk
            return

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
            cached_content=cached_content or self.cached_content,
            tool_choice=tool_choice,
            **kwargs,
        )
        prev_usage_metadata: UsageMetadata | None = None  # Cumulative usage
        indexer = _StreamBlockIndexer()
        stream = await self.async_client.models.generate_content_stream(
            **request,
        )
        gc_stream = _aclassified_stream(stream, request)
        if (active_gc_sink := _ACTIVE_ASTREAM_SINK.get()) is not None:
            active_gc_sink.append(gc_stream)
        async with _safe_aclosing(gc_stream) as managed_gc_stream:
            saw_tool_call = False
            async for chunk in managed_gc_stream:
                _chat_result = _response_to_result(
                    chunk, stream=True, prev_usage=prev_usage_metadata
                )
                gen = cast("ChatGenerationChunk", _chat_result.generations[0])
                message = cast("AIMessageChunk", gen.message)

                # populate index if missing
                indexer.assign(message)
                saw_tool_call = _mark_streamed_tool_call(gen, saw_tool_call)

                prev_usage_metadata = (
                    message.usage_metadata
                    if prev_usage_metadata is None
                    else add_usage(prev_usage_metadata, message.usage_metadata)
                )

                if run_manager:
                    await run_manager.on_llm_new_token(gen.text, chunk=gen)
                yield gen

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text. Uses the model's tokenizer.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.

        Example:
            ```python
            llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
            num_tokens = llm.get_num_tokens("Hello, world!")
            print(num_tokens)
            # -> 4
            ```
        """
        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)

        result = self.client.models.count_tokens(
            model=self.model, contents=[Content(parts=[Part(text=text)])]
        )
        return result.total_tokens if result and result.total_tokens is not None else 0

    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        method: Literal["function_calling", "json_mode", "json_schema"]
        | None = "json_schema",
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Return a `Runnable` that constrains model output to a given schema.

        Constrains the model to return output conforming to the provided schema.

        Supports Pydantic models, `TypedDict`, and JSON schema dictionaries.

        Args:
            schema: The output schema as a Pydantic `BaseModel` class, a `TypedDict`
                class, or a JSON schema dictionary.
            method: The method to use for structured output.

                Options:

                - `'json_schema'` (recommended): Uses native JSON schema support for
                    reliable structured output. Supports streaming with fully-parsed
                    Pydantic objects.
                - `'json_mode'`: Deprecated alias for `'json_schema'`.
                - `'function_calling'`: Uses tool/function calling. Less reliable than
                    `'json_schema'` and not recommended for new code.
            include_raw: If `True`, returns a dict with both the raw model output
                and the parsed structured output.

        Returns:
            A `Runnable` that takes the same input as the chat model but returns the
                structured output. When streaming, emits fully-parsed objects of the
                specified schema type (not incremental JSON strings).

        Example:
            ```python title="Basic usage with Pydantic model"
            from pydantic import BaseModel
            from langchain_google_genai import ChatGoogleGenerativeAI


            class Person(BaseModel):
                name: str
                age: int


            model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
            structured_model = model.with_structured_output(
                Person,
                method="json_schema",
            )

            result = structured_model.invoke(
                "Tell me about a person named Alice, age 30"
            )
            print(result)  # Person(name="Alice", age=30)
            ```

            ```python title="Streaming structured output"
            from pydantic import BaseModel
            from langchain_google_genai import ChatGoogleGenerativeAI


            class Recipe(BaseModel):
                name: str
                ingredients: list[str]
                steps: list[str]


            model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
            structured_model = model.with_structured_output(
                Recipe, method="json_schema"
            )

            # Emits fully-parsed Recipe objects, not incremental JSON strings
            for chunk in structured_model.stream(
                "Give me a recipe for chocolate chip cookies"
            ):
                print(chunk)  # Recipe(name=..., ingredients=[...], steps=[...])
            ```

            ```python title="Using with dict schema"
            model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

            schema = {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "integer"},
                },
                "required": ["title", "priority"],
            }

            structured_model = model.with_structured_output(
                schema, method="json_schema"
            )
            result = structured_model.invoke("Create a task: finish report, priority 1")
            print(result)  # {"title": "finish report", "priority": 1}
            ```

            ```python title="Including raw output"
            structured_model = model.with_structured_output(
                Person, method="json_schema", include_raw=True
            )

            result = structured_model.invoke("Tell me about Bob, age 25")
            print(result["parsed"])  # Person(name="Bob", age=25)
            print(result["raw"])  # AIMessage with full model response
            ```
        """
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        parser: OutputParserLike
        llm: Runnable[LanguageModelInput, AIMessage]

        # `json_mode` kept for backwards compatibility; shouldn't be used in new code
        if method in ("json_mode", "json_schema"):
            # For ls_structured_output_format, we use convert_to_json_schema for
            # Pydantic/TypedDict/dict-with-title schemas to match langchain standard
            # tests. Raw dicts without titles are passed through as-is.
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                # Handle Pydantic models
                if issubclass(schema, BaseModelV1):
                    # Use legacy schema generation for pydantic v1 models
                    schema_json = schema.schema()
                else:
                    schema_json = schema.model_json_schema()
                parser = PydanticOutputParser(pydantic_object=schema)
                ls_schema = convert_to_json_schema(schema)
            elif is_typeddict(schema):
                schema_json = convert_to_json_schema(schema)
                parser = JsonOutputParser()
                ls_schema = schema_json
            elif isinstance(schema, dict):
                schema_json = schema
                parser = JsonOutputParser()
                # Dicts with title can be converted; raw dicts pass through as-is
                ls_schema = (
                    convert_to_json_schema(schema) if "title" in schema else schema
                )
            else:
                msg = f"Unsupported schema type {type(schema)}"
                raise ValueError(msg)

            # Note: The Google GenAI SDK automatically handles schema transformation
            # (inlining $defs, resolving $ref) via its process_schema() function.
            # This ensures Union types and nested schemas work correctly.
            bind_so_kwargs: dict[str, Any] = {}
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                bind_so_kwargs["_structured_output_schema"] = schema
            llm = self.bind(
                response_mime_type="application/json",
                response_json_schema=schema_json,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": ls_schema,
                },
                **bind_so_kwargs,
            )
        else:
            # LangChain tool calling structured output method (discouraged)
            tool_name = _get_tool_name(schema)  # type: ignore[arg-type]
            if isinstance(schema, type) and is_basemodel_subclass_safe(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
            tool_choice = tool_name if self._supports_tool_choice else None
            try:
                llm = self.bind_tools(
                    [schema],
                    tool_choice=tool_choice,
                    ls_structured_output_format={
                        "kwargs": {"method": "function_calling"},
                        "schema": convert_to_openai_tool(schema),
                    },
                )
            except Exception:
                llm = self.bind_tools([schema], tool_choice=tool_choice)
        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback
        return llm | parser

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type | Callable[..., Any] | BaseTool | GoogleTool
        ],
        tool_config: dict | ToolConfig | None = None,
        *,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Can be a pydantic model, `Callable`, or `BaseTool`. Pydantic models,
                `Callable`, and `BaseTool` objects will be automatically converted to
                their schema dictionary representation.

                Tools with Union types in their arguments are now supported and
                converted to `anyOf` schemas.
            tool_config: Optional tool configuration for additional settings like
                `retrieval_config` (for Google Maps/Google Search grounding).

                Can be used together with `tool_choice`, but cannot specify
                `function_calling_config` in `tool_config` if `tool_choice` is also
                provided (they would conflict).

                !!! example "Example with Google Maps grounding"

                    ```python
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")

                    response = model.invoke(
                        "What Italian restaurants are near here?",
                        tools=[{"google_maps": {}}],
                        tool_choice="required",
                        tool_config={
                            "retrieval_config": {
                                "lat_lng": {
                                    "latitude": 48.858844,
                                    "longitude": 2.294351,
                                }
                            }
                        },
                    )
                    ```
            tool_choice: Control how the model uses tools.

                Options:

                - `'auto'` (default): Model decides whether to call functions
                - `'any'` or `'required'`: Model must call a function (both are
                    equivalent)
                - `'none'`: Model cannot call functions
                - `'function_name'`: Model must call the specified function
                - `['fn1', 'fn2']`: Model must call one of the specified functions
                - `True`: Same as `'any'`

                Can be used together with `tool_config` to control function calling
                while also providing additional configuration like `retrieval_config`.
            **kwargs: Any additional parameters to pass to the `Runnable` constructor.
        """
        formatted_tools: list[Any] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                mcp_shorthand = tool.get("mcp_servers") or tool.get("mcpServers")
                is_ia_mcp_shorthand = (
                    set(tool) in ({"mcp_servers"}, {"mcpServers"})
                    and isinstance(mcp_shorthand, list)
                    and any(
                        isinstance(srv, Mapping)
                        and any(
                            k in srv
                            for k in (
                                "url",
                                "streamable_http_url",
                                "headers",
                                "allowed_tools",
                            )
                        )
                        for srv in mcp_shorthand
                    )
                )
                if (
                    tool.get("type") in _INTERACTIONS_SERVER_TOOL_TYPES
                    or (tool.get("type") == "function" and "name" in tool)
                    or is_ia_mcp_shorthand
                ):
                    formatted_tools.append(copy.deepcopy(dict(tool)))
                elif (
                    "name" in tool
                    and "input_schema" in tool
                    and "parameters" not in tool
                ):
                    params_in = copy.deepcopy(dict(tool["input_schema"]))
                    params_in.setdefault("type", "object")
                    params_in.setdefault("properties", {})
                    fn_obj: dict[str, Any] = {
                        "name": str(tool["name"]),
                        "parameters": params_in,
                    }
                    if tool.get("description"):
                        fn_obj["description"] = str(tool["description"])
                    formatted_tools.append({"type": "function", "function": fn_obj})
                else:
                    try:
                        formatted_tools.append(
                            convert_to_openai_tool(copy.deepcopy(dict(tool)))
                        )
                    except Exception:
                        formatted_tools.extend(
                            tool_to_dict(t)
                            for t in convert_to_genai_function_declarations([tool])
                        )
            elif isinstance(tool, Runnable) and not isinstance(tool, BaseTool):
                formatted_tools.append(convert_to_openai_tool(tool.as_tool()))
            elif isinstance(tool, GoogleTool):
                formatted_tools.append(tool_to_dict(tool))
            elif isinstance(tool, FunctionDeclaration):
                dumped_fd = tool.model_dump(exclude_none=True)
                if dumped_fd.get("parameters_json_schema") is not None:
                    fn_fd: dict[str, Any] = {
                        "name": str(dumped_fd["name"]),
                        "parameters": copy.deepcopy(
                            dumped_fd["parameters_json_schema"]
                        ),
                    }
                    if dumped_fd.get("description"):
                        fn_fd["description"] = str(dumped_fd["description"])
                    formatted_tools.append({"type": "function", "function": fn_fd})
                else:
                    formatted_tools.extend(
                        tool_to_dict(t)
                        for t in convert_to_genai_function_declarations([tool])
                    )
            else:
                try:
                    formatted_tools.append(convert_to_openai_tool(tool))
                except Exception:
                    formatted_tools.extend(
                        tool_to_dict(t)
                        for t in convert_to_genai_function_declarations([tool])
                    )
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if tool_config:
            kwargs["tool_config"] = tool_config
        return self.bind(tools=formatted_tools, **kwargs)

    @property
    def _supports_tool_choice(self) -> bool:
        """Whether the model supports the `tool_choice` parameter.

        See the [Gemini models docs](https://ai.google.dev/gemini-api/docs/models) for a
        full list. Gemini calls this "function calling".
        """
        return self.profile.get("tool_choice", True) if self.profile else True


def _get_tool_name(
    tool: _ToolDict | GoogleTool | dict,
) -> str:
    try:
        genai_tools = convert_to_genai_function_declarations([tool])
        genai_tool = tool_to_dict(genai_tools[0])
        return next(f["name"] for f in genai_tool["function_declarations"])  # type: ignore[index]
    except ValueError:  # other TypedDict
        if is_typeddict(tool):
            return convert_to_openai_tool(cast("dict", tool))["function"]["name"]
        raise
