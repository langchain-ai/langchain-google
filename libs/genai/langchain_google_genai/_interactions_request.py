"""Request preparation and message-to-step translation for Gemini Interactions API."""

from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import json
import logging
import mimetypes
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import filetype  # type: ignore[import-untyped]
from google.genai.types import HttpOptions, ToolConfig
from google.genai.types import Tool as GoogleTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import is_data_content_block

from langchain_google_genai._common import SafetySettingDict
from langchain_google_genai._compat import (
    _MEDIA_PROCESSING_TOOL_NAME,
    _REASONING_BLOCK_TYPES,
    _classify_model_provider,
    _ModelProviderKind,
)
from langchain_google_genai._function_utils import _ToolChoiceType
from langchain_google_genai._image_utils import ImageBytesLoader, Route
from langchain_google_genai._interactions_tools import (
    _HAS_NATIVE_CONTINUATION_TOKEN,
    _build_interactions_generation_config,
    _format_interactions_agent_config,
    _format_interactions_response_format,
    _format_interactions_safety_settings,
    _format_interactions_speech_config,
    _format_interactions_tool_choice,
    _format_interactions_tools,
    _resolve_interactions_thinking,
    _unwrap_enum_value,
)
from langchain_google_genai.chat_models import (
    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY,
    _HTTPX_ASYNC_CLIENT_TYPES,
    _HTTPX_ASYNC_TRANSPORT_TYPES,
    _HTTPX_REQUEST_ERRORS,
    _HTTPX_SYNC_CLIENT_TYPES,
    _HTTPX_SYNC_TRANSPORT_TYPES,
    _HTTPX_TIMEOUT_ERRORS,
    _HTTPX_TIMEOUT_TYPES,
    _HTTPX_UNSUPPORTED_PROTOCOL_ERRORS,
    _INTERACTIONS_AUTO_MODEL_PREFIXES,
    _PROVIDER_RESOLVED_HOSTS,
    ChatGoogleGenerativeAIError,
    GoogleConnectionError,
    GoogleInvalidRequestError,
    GoogleTimeoutError,
    _block_signature,
    _create_httpx_client,
    _function_call_signatures_from_content,
    _is_gemini_3_or_later,
    _is_gemini_25_model,
    _is_rebuilt_tool_call_block,
    _merge_ephemeral_httpx_kwargs,
    _normalize_gemini_model_name,
    _uses_fixed_sampling_and_disallows_prefill,
)

logger = logging.getLogger(__name__)

_DEFAULT_INTERACTIONS_TIMEOUT_S = 600.0
_DEFAULT_POLLING_INTERVAL_S = 2.0
_SKIP_THOUGHT_SIGNATURE_B64 = "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I="

_AUDIO_FORMAT_MAP: dict[str, str] = {
    "wav": "audio/wav",
    "wave": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "mp3": "audio/mp3",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "l16": "audio/l16",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "aiff": "audio/aiff",
    "m4a": "audio/m4a",
    "mpeg": "audio/mpeg",
    "opus": "audio/opus",
    "alaw": "audio/alaw",
    "mulaw": "audio/mulaw",
    "webm": "audio/webm",
}

_RAW_SERVER_STEP_TYPES = frozenset(
    {
        "code_execution_call",
        "code_execution_result",
        "google_search_call",
        "google_search_result",
        "url_context_call",
        "url_context_result",
        "google_maps_call",
        "google_maps_result",
        "file_search_call",
        "file_search_result",
        "mcp_server_tool_call",
        "mcp_server_tool_result",
        "retrieval_call",
        "retrieval_result",
    }
)


@dataclass
class _InteractionsPreparedRequest:
    """Prepared request state passed to sync/async Interactions executors."""

    create_kwargs: dict[str, Any]
    target_name: str
    max_retries: int | None
    requested_store: bool
    background: bool
    polling_interval: float
    timeout_seconds: float | None
    api_version: str | None
    extra_headers: dict[str, str] | None
    output_version: str | None
    pydantic_schema: type | None
    poll: bool = True
    previous_interaction_id: str | None = None
    base_url_override: str | None = None
    client_args_override: dict[str, Any] | None = None
    async_client_args_override: dict[str, Any] | None = None
    httpx_client_override: Any = None
    httpx_async_client_override: Any = None


def _strip_model_prefix(raw_model: str) -> str:
    """Strip leading `'models/'` or `'publishers/google/models/'` prefixes."""
    if "/models/" in raw_model:
        return raw_model.rsplit("/models/", 1)[-1]
    if raw_model.startswith("models/"):
        return raw_model[len("models/") :]
    if raw_model.startswith("agents/"):
        return raw_model[len("agents/") :]
    return raw_model


def _normalize_signature_b64(sig: Any) -> str | None:
    """Return a single-encoded base64 signature string, never double-encoding `str`."""
    if isinstance(sig, bytes):
        return base64.b64encode(sig).decode("ascii") if sig else None
    if isinstance(sig, str):
        cleaned = sig.strip()
        if not cleaned:
            return None
        if cleaned == "skip_thought_signature_validator":
            return _SKIP_THOUGHT_SIGNATURE_B64
        return cleaned
    return None


def _extract_trusted_signature(
    block: Mapping[str, Any], *, provider_kind: _ModelProviderKind
) -> str | None:
    """Return a base64 signature only when known to originate from Gemini.

    Unstamped `AIMessage` objects (`provider_kind == "unknown"`) may carry
    Anthropic `thinking.signature` blobs in cross-provider message histories,
    which Gemini rejects with HTTP 400. Generic `signature` fields are therefore
    replayed only when `model_provider` is a native Google provider, while
    explicit Gemini keys (`thought_signature` and the official bypass sentinel)
    remain honored on unstamped messages.
    """
    if provider_kind == "foreign":
        return None
    if provider_kind == "native":
        return _normalize_signature_b64(_block_signature(block))
    explicit_gemini_sig = block.get("thought_signature")
    if explicit_gemini_sig:
        return _normalize_signature_b64(explicit_gemini_sig)
    norm_generic = _normalize_signature_b64(_block_signature(block))
    if norm_generic == _SKIP_THOUGHT_SIGNATURE_B64:
        return _SKIP_THOUGHT_SIGNATURE_B64
    return None


def _get_last_interaction_messages(
    messages: Sequence[BaseMessage],
    *,
    target_id: str | None = None,
) -> tuple[str | None, list[BaseMessage]]:
    """Scan backward for a stored native Google `AIMessage` and slice trailing items.

    When `target_id` is specified (explicit `previous_interaction_id`), locates the
    matching non-foreign `AIMessage` for history branching. Otherwise requires
    `_classify_model_provider == "native"` and `stored is True` so foreign or
    unstamped provider IDs are never sent to Gemini.
    """
    clean_target = target_id.strip() if isinstance(target_id, str) else None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, AIMessage):
            continue
        meta = msg.response_metadata or {}
        provider_kind = _classify_model_provider(meta.get("model_provider"))
        if clean_target is not None:
            if provider_kind == "foreign":
                continue
            cand_iid = meta.get("interaction_id")
            cand_id = meta.get("id")
            if (isinstance(cand_iid, str) and cand_iid.strip() == clean_target) or (
                isinstance(cand_id, str) and cand_id.strip() == clean_target
            ):
                return clean_target, list(messages[idx + 1 :])
            continue

        if provider_kind != "native" or meta.get("stored") is not True:
            continue
        iid = meta.get("interaction_id") or meta.get("id")
        if isinstance(iid, str) and iid.strip():
            return iid.strip(), list(messages[idx + 1 :])
    return None, list(messages)


def _resolve_explicit_store(
    *,
    call_store: Any,
    model_kwargs_store: Any,
    instance_store: Any,
) -> bool | None:
    """Return caller/instance explicit `store` bool override, or `None` if unset."""
    if isinstance(call_store, bool):
        return call_store
    if isinstance(model_kwargs_store, bool):
        return model_kwargs_store
    if isinstance(instance_store, bool):
        return instance_store
    return None


def _resolve_store(
    *,
    call_store: bool | None,
    model_kwargs_store: bool | None,
    instance_store: bool | None,
    use_previous_interaction_id: bool,
    previous_interaction_id: str | None,
    background: bool,
    agent: str | None,
) -> bool:
    """Resolve smart `store` default and validate incompatible combinations."""
    explicit = _resolve_explicit_store(
        call_store=call_store,
        model_kwargs_store=model_kwargs_store,
        instance_store=instance_store,
    )

    if explicit is not None:
        resolved = explicit
    elif (
        use_previous_interaction_id
        or bool(previous_interaction_id)
        or background
        or bool(agent)
    ):
        resolved = True
    else:
        resolved = False

    if resolved is False and (previous_interaction_id or background):
        msg = (
            "store=False cannot be combined with previous_interaction_id "
            "or background=True in the Gemini Interactions API."
        )
        raise GoogleInvalidRequestError(msg)
    return resolved


def _clean_continuation_token(raw_token: Any) -> str | None:
    """Normalize an optional `continuation_token` to a non-empty string or `None`."""
    if raw_token is None:
        return None
    cleaned = str(raw_token).strip()
    return cleaned or None


def _resolve_stateful_slicing(
    messages: Sequence[BaseMessage],
    *,
    use_prev_flag: bool,
    explicit_prev_id: Any,
    explicit_store: bool | None,
    continuation_token: str | None = None,
) -> tuple[str | None, list[BaseMessage] | None, list[BaseMessage]]:
    """Resolve `(prev_id, prior_messages, active_messages)` respecting `store=False`."""
    prior_msgs: list[BaseMessage] | None = None
    active_messages = list(messages)
    prev_id: str | None = (
        str(explicit_prev_id).strip()
        if explicit_prev_id and str(explicit_prev_id).strip()
        else None
    )
    clean_cont = _clean_continuation_token(continuation_token)

    if prev_id is not None:
        matched_id, matched_tail = _get_last_interaction_messages(
            messages, target_id=prev_id
        )
        if matched_id == prev_id and matched_tail:
            cut = len(messages) - len(matched_tail)
            prior_msgs = list(messages[:cut])
            active_messages = matched_tail
        elif clean_cont is not None and explicit_store is not False:
            if matched_id == prev_id and not matched_tail:
                prior_msgs = list(messages)
                active_messages = []
            elif (
                matched_id is None and messages and isinstance(messages[-1], AIMessage)
            ):
                last_meta = messages[-1].response_metadata or {}
                last_kind = _classify_model_provider(last_meta.get("model_provider"))
                last_iid = last_meta.get("interaction_id") or last_meta.get("id")
                last_tok = _clean_continuation_token(
                    last_meta.get("continuation_token")
                )
                if last_kind != "foreign" and (
                    not (isinstance(last_iid, str) and last_iid.strip())
                    or last_iid.strip() == prev_id
                    or last_tok == clean_cont
                ):
                    prior_msgs = list(messages)
                    active_messages = []
    elif explicit_store is not False:
        if use_prev_flag:
            sliced_id, sliced_msgs = _get_last_interaction_messages(messages)
            if sliced_id and sliced_msgs:
                prev_id = sliced_id
                cut = len(messages) - len(sliced_msgs)
                prior_msgs = list(messages[:cut])
                active_messages = sliced_msgs
            elif sliced_id and not sliced_msgs and clean_cont is not None:
                prev_id = sliced_id
                prior_msgs = list(messages)
                active_messages = []
            elif sliced_id and not sliced_msgs and len(messages) > 1:
                # Latest stored AIMessage was trailing; try earlier messages
                earlier_id, earlier_tail = _get_last_interaction_messages(messages[:-1])
                if earlier_id and earlier_tail:
                    prev_id = earlier_id
                    cut = (len(messages) - 1) - len(earlier_tail)
                    prior_msgs = list(messages[:cut])
                    active_messages = [*earlier_tail, messages[-1]]
        elif (
            clean_cont is not None and messages and isinstance(messages[-1], AIMessage)
        ):
            last_meta = messages[-1].response_metadata or {}
            last_kind = _classify_model_provider(last_meta.get("model_provider"))
            last_iid = last_meta.get("interaction_id") or last_meta.get("id")
            last_tok = _clean_continuation_token(last_meta.get("continuation_token"))
            if (
                last_kind != "foreign"
                and last_meta.get("stored") is True
                and isinstance(last_iid, str)
                and last_iid.strip()
                and (last_tok is None or last_tok == clean_cont)
            ):
                prev_id = last_iid.strip()
                prior_msgs = list(messages)
                active_messages = []

    return prev_id, prior_msgs, active_messages


def _normalize_audio_mime(raw: str | None) -> str:
    """Normalize shorthand audio format tags and MIME aliases."""
    if not raw:
        return "audio/wav"
    lowered = raw.strip().lower()
    if lowered in _AUDIO_FORMAT_MAP:
        return _AUDIO_FORMAT_MAP[lowered]
    if lowered.startswith("audio/"):
        return lowered
    return f"audio/{lowered}"


def _normalize_resolution(raw_res: Any, *, model: str | None) -> str | None:
    """Normalize per-part `media_resolution` for Gemini 3+ models."""
    if raw_res is None:
        return None
    if model and _is_gemini_25_model(model):
        warnings.warn(
            "Setting per-part media resolution requests to Gemini 2.5 models and "
            "older is not supported. The media_resolution parameter will be ignored.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if model and not _is_gemini_3_or_later(model):
        return None
    val = str(_unwrap_enum_value(raw_res)).lower()
    val = val.removeprefix("media_resolution_")
    return val if val and val != "unspecified" else None


def _normalize_processing(raw_proc: Any) -> str | dict[str, Any] | None:
    """Normalize `media_processing` / `processing` onto `'agentic' | 'static'`."""
    if raw_proc is None:
        return None
    if isinstance(raw_proc, Mapping):
        return copy.deepcopy(dict(raw_proc))
    val = str(_unwrap_enum_value(raw_proc)).lower()
    val = val.removeprefix("media_processing_")
    return val or None


def _is_data_uri(uri: str) -> bool:
    """Return True when `uri` uses a case-insensitive `data:` scheme."""
    return uri.lstrip().lower().startswith("data:")


def _is_gs_uri(uri: str) -> bool:
    """Return True when `uri` uses a case-insensitive `gs://` scheme."""
    return uri.lstrip().lower().startswith("gs://")


def _is_files_api_id(uri: str) -> bool:
    """Return True when `uri` is a Gemini Files API identifier (`files/...`)."""
    stripped = uri.strip()
    return (
        stripped.startswith("files/")
        and not os.path.exists(stripped)
        and not os.path.exists(uri)
    )


def _has_http_scheme(url: str) -> bool:
    """Return True when `url` has a case-insensitive `http` or `https` scheme + host."""
    parsed = urlparse(url)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _canonicalize_uri_scheme(uri: str) -> str:
    """Lowercase leading `http://`, `https://`, or `gs://`, preserving path case."""
    stripped = uri.strip()
    return re.sub(
        r"^(https?|gs)://",
        lambda m: f"{m.group(1).lower()}://",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    )


def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(host == h or host.endswith(f".{h}") for h in ("youtube.com", "youtu.be"))


def _is_provider_uri(url: str) -> bool:
    if _is_gs_uri(url):
        return True
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(host == h or host.endswith(f".{h}") for h in _PROVIDER_RESOLVED_HOSTS)


def _finalize_fetched_media_bytes(
    url: str,
    raw_bytes: bytes,
    raw_content_type: str | None,
) -> tuple[str, str | None]:
    """Infer MIME type from headers/path/magic bytes and base64-encode `raw_bytes`."""
    parsed = urlparse(url)
    content_type = (
        raw_content_type.split(";", 1)[0].strip().lower() if raw_content_type else None
    )
    guessed_mime = content_type
    if not guessed_mime or guessed_mime == "application/octet-stream":
        guessed_mime, _ = mimetypes.guess_type(parsed.path or url)
    if not guessed_mime:
        kind = filetype.guess(raw_bytes)
        if kind:
            guessed_mime = kind.mime
    return base64.b64encode(raw_bytes).decode("ascii"), guessed_mime


_TEXT_CONTENT_BLOCK_KEYS = frozenset(
    {
        "type",
        "text",
        "id",
        "index",
        "annotations",
        "citations",
        "extras",
        "cache_control",
    }
)


def _validate_external_media_url(url: str, image_loader: ImageBytesLoader) -> str:
    """Enforce `_image_utils` local-file/scheme validation and return normalized URL."""
    if os.path.exists(url):
        image_loader.load_bytes(url)
    cleaned = url.strip()
    route = image_loader._route(cleaned)
    if route != Route.URL:
        # Triggers the standard local-file / invalid-URI error from ImageBytesLoader
        image_loader.load_bytes(cleaned)
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        msg = f"Unsupported media URL scheme '{parsed.scheme}'; expected http or https."
        raise ValueError(msg)
    return _canonicalize_uri_scheme(cleaned)


def _is_tool_result_content_block(item: Any) -> bool:
    """Return True only when a tool-result list element is a real content/media block.

    Prevents domain JSON dictionaries returned by tools (such as
    `{"id": "tx_1", "type": "credit", "amount": 50.0}` or
    `{"type": "image", "name": "ubuntu"}`) from being misinterpreted as LangChain
    multimodal blocks.
    """
    if not isinstance(item, Mapping):
        return False
    if set(item.keys()) == {"cachePoint"} or item.get("type") == "cache_point":
        return True
    btype = item.get("type")
    if not isinstance(btype, str):
        return False
    if btype == "non_standard":
        inner = item.get("value")
        return isinstance(inner, Mapping) and _is_tool_result_content_block(inner)
    if btype == "text":
        return isinstance(item.get("text"), str) and set(item.keys()).issubset(
            _TEXT_CONTENT_BLOCK_KEYS
        )
    if btype == "text-plain":
        return (
            isinstance(item.get("text"), str)
            or isinstance(item.get("base64"), (str, bytes, os.PathLike))
            or isinstance(item.get("data"), (str, bytes, os.PathLike))
            or (isinstance(item.get("url"), str) and bool(item.get("url")))
        )
    if btype == "image_url":
        raw_iu = item.get("image_url")
        return (isinstance(raw_iu, str) and bool(raw_iu)) or (
            isinstance(raw_iu, Mapping) and isinstance(raw_iu.get("url"), str)
        )
    if btype == "input_audio":
        ia = item.get("input_audio")
        return isinstance(ia, Mapping) and "data" in ia
    if btype in ("image", "document") and isinstance(item.get("source"), Mapping):
        src = item["source"]
        return src.get("type") in ("base64", "url", "text")
    if btype == "file" and isinstance(item.get("file"), Mapping):
        fobj = item["file"]
        return bool(fobj.get("file_data") or fobj.get("file_id"))
    if btype in ("media", "file_data"):
        return ("data" in item and item["data"] is not None) or (
            isinstance(item.get("file_uri"), str) and bool(item.get("file_uri"))
        )
    if is_data_content_block(dict(item)):
        return True
    if btype in ("image", "audio", "video", "document", "file"):
        return (
            isinstance(item.get("base64"), (str, bytes, os.PathLike))
            or isinstance(item.get("data"), (str, bytes, os.PathLike))
            or (isinstance(item.get("uri"), str) and bool(item.get("uri")))
            or (isinstance(item.get("url"), str) and bool(item.get("url")))
            or (isinstance(item.get("file_id"), str) and bool(item.get("file_id")))
        )
    return False


def _extract_inbound_media_url(part: Mapping[str, Any]) -> str | None:
    """Extract the exact media URL `_convert_block_to_interaction_content` consumes."""
    if set(part.keys()) == {"cachePoint"}:
        return None
    btype = part.get("type")
    if btype == "non_standard":
        inner = part.get("value")
        return _extract_inbound_media_url(inner) if isinstance(inner, Mapping) else None
    if btype == "text-plain":
        if "text" in part and isinstance(part["text"], str):
            return None
        if part.get("base64") or part.get("data"):
            return None
        url_tp = part.get("url")
        return url_tp if isinstance(url_tp, str) and url_tp else None
    if btype == "image_url":
        raw_iu = part.get("image_url")
        if isinstance(raw_iu, Mapping):
            u_iu = raw_iu.get("url")
            return u_iu if isinstance(u_iu, str) and u_iu else None
        return raw_iu if isinstance(raw_iu, str) and raw_iu else None
    if btype in ("image", "document") and isinstance(part.get("source"), Mapping):
        src = part["source"]
        if src.get("type") == "url" and isinstance(src.get("url"), str):
            return src["url"] or None
        return None
    if btype == "file" and isinstance(part.get("file"), Mapping):
        fobj = part["file"]
        fdata = fobj.get("file_data")
        if isinstance(fdata, str) and fdata:
            return fdata if _has_http_scheme(fdata) else None
        fid = fobj.get("file_id")
        return fid if isinstance(fid, str) and fid else None
    if btype in ("media", "file_data"):
        if "data" in part and part["data"] is not None:
            return None
        f_uri = part.get("file_uri")
        return f_uri if isinstance(f_uri, str) and f_uri else None
    if btype in ("image", "audio", "video", "document", "file"):
        uri_cand = part.get("uri") or part.get("file_id")
        if not uri_cand and (part.get("source_type") == "url" or "url" in part):
            uri_cand = part.get("url")
        return uri_cand if isinstance(uri_cand, str) and uri_cand else None
    return None


def _validate_media_response(url: str, resp: Any) -> tuple[str, str | None]:
    """Validate redirect schemes + HTTP status and encode fetched media bytes."""
    for hop in getattr(resp, "history", None) or ():
        hop_url = getattr(hop, "url", None)
        if hop_url is not None:
            raw_hop_s = (
                urlparse(hop_url).scheme
                if isinstance(hop_url, str)
                else str(getattr(hop_url, "scheme", ""))
            )
            hop_scheme = raw_hop_s.lower()
            if hop_scheme and hop_scheme not in {"http", "https"}:
                msg = f"Disallowed redirect media URL scheme '{hop_scheme}'."
                raise ValueError(msg)
    resp_url = getattr(resp, "url", None)
    if resp_url is not None:
        raw_s = (
            urlparse(resp_url).scheme
            if isinstance(resp_url, str)
            else str(getattr(resp_url, "scheme", ""))
        )
        resp_scheme = raw_s.lower()
        if resp_scheme and resp_scheme not in {"http", "https"}:
            msg = f"Disallowed redirect media URL scheme '{resp_scheme}'."
            raise ValueError(msg)
    resp.raise_for_status()
    return _finalize_fetched_media_bytes(
        url, resp.content, resp.headers.get("Content-Type")
    )


_MEDIA_FETCH_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    *_HTTPX_TIMEOUT_ERRORS,
)


def _fetch_with_sync_client(
    url: str,
    client: Any,
    image_loader: ImageBytesLoader,
    *,
    request_timeout: Any = None,
    enforce_follow_redirects: bool = False,
) -> tuple[str, str | None]:
    """Download an external HTTP(S) URL via an `httpx2`/`httpx` sync client."""
    target_url = _validate_external_media_url(url, image_loader)
    test_ua = os.environ.get("LANGCHAIN_TESTS_USER_AGENT")
    get_kwargs: dict[str, Any] = {}
    if test_ua:
        get_kwargs["headers"] = {"User-Agent": test_ua}
    if request_timeout is not None:
        get_kwargs["timeout"] = request_timeout
    if enforce_follow_redirects:
        get_kwargs["follow_redirects"] = True
    try:
        resp = client.get(target_url, **get_kwargs)
    except _MEDIA_FETCH_TIMEOUT_ERRORS as exc:
        msg_to = f"Timed out fetching external media URL '{url}': {exc}"
        raise GoogleTimeoutError(msg_to) from exc
    except _HTTPX_UNSUPPORTED_PROTOCOL_ERRORS as exc:
        msg_proto = (
            f"Disallowed redirect media URL scheme while fetching '{url}': {exc}"
        )
        raise ValueError(msg_proto) from exc
    except _HTTPX_REQUEST_ERRORS as exc:
        err_url = getattr(getattr(exc, "request", None), "url", None)
        err_scheme = str(getattr(err_url, "scheme", "")).lower() if err_url else ""
        if err_scheme and err_scheme not in {"http", "https"}:
            msg_s = f"Disallowed redirect media URL scheme '{err_scheme}'."
            raise ValueError(msg_s) from exc
        msg_conn = f"Connection error fetching external media URL '{url}': {exc}"
        raise GoogleConnectionError(msg_conn) from exc
    return _validate_media_response(target_url, resp)


def _fetch_external_url(
    url: str,
    image_loader: ImageBytesLoader,
    *,
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
) -> tuple[str, str | None]:
    """Download an HTTP(S) URL securely with bounded timeout and Content-Type MIME."""
    if prefetched_urls and url in prefetched_urls:
        return prefetched_urls[url]

    with _create_httpx_client(
        {"timeout": 60.0, "follow_redirects": True}, is_async=False
    ) as http_client:
        return _fetch_with_sync_client(url, http_client, image_loader)


async def _afetch_external_url(
    url: str,
    client: Any,
    image_loader: ImageBytesLoader,
    *,
    request_timeout: Any = None,
    enforce_follow_redirects: bool = False,
) -> tuple[str, tuple[str, str | None]]:
    """Download an external HTTP(S) URL asynchronously without blocking asyncio."""
    target_url = _validate_external_media_url(url, image_loader)
    test_ua = os.environ.get("LANGCHAIN_TESTS_USER_AGENT")
    get_kwargs: dict[str, Any] = {}
    if test_ua:
        get_kwargs["headers"] = {"User-Agent": test_ua}
    if request_timeout is not None:
        get_kwargs["timeout"] = request_timeout
    if enforce_follow_redirects:
        get_kwargs["follow_redirects"] = True
    try:
        resp = await client.get(target_url, **get_kwargs)
    except _MEDIA_FETCH_TIMEOUT_ERRORS as exc:
        msg_to = f"Timed out fetching external media URL '{url}': {exc}"
        raise GoogleTimeoutError(msg_to) from exc
    except _HTTPX_UNSUPPORTED_PROTOCOL_ERRORS as exc:
        msg_proto = (
            f"Disallowed redirect media URL scheme while fetching '{url}': {exc}"
        )
        raise ValueError(msg_proto) from exc
    except _HTTPX_REQUEST_ERRORS as exc:
        err_url = getattr(getattr(exc, "request", None), "url", None)
        err_scheme = str(getattr(err_url, "scheme", "")).lower() if err_url else ""
        if err_scheme and err_scheme not in {"http", "https"}:
            msg_s = f"Disallowed redirect media URL scheme '{err_scheme}'."
            raise ValueError(msg_s) from exc
        msg_conn = f"Connection error fetching external media URL '{url}': {exc}"
        raise GoogleConnectionError(msg_conn) from exc
    return url, _validate_media_response(target_url, resp)


def _collect_external_fetch_urls(messages: Sequence[BaseMessage]) -> list[str]:
    """Collect external HTTP(S) media URLs strictly from inbound user/tool media blocks.

    Inspects only `HumanMessage`, `ToolMessage`, and `FunctionMessage` multimodal
    content blocks — never `AIMessage`, `SystemMessage`, tool call arguments, server
    tool calls/results, citations, or arbitrary domain JSON dictionaries.
    """
    found: list[str] = []
    seen: set[str] = set()
    image_loader = ImageBytesLoader()

    def _maybe_record(cand: str | None) -> None:
        if not cand or not isinstance(cand, str):
            return
        if _is_files_api_id(cand) or _is_gs_uri(cand) or _is_data_uri(cand):
            return
        if not _has_http_scheme(cand):
            return
        if _is_youtube_url(cand) or _is_provider_uri(cand):
            return
        if (
            cand not in seen
            and not os.path.exists(cand)
            and image_loader._route(cand.strip()) == Route.URL
        ):
            seen.add(cand)
            found.append(cand)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if not isinstance(msg.content, list):
                continue
            for h_item in msg.content:
                if not isinstance(h_item, Mapping):
                    continue
                if h_item.get("type") == "tool_result":
                    tr_content = h_item.get("content")
                    if isinstance(tr_content, list):
                        for sub in tr_content:
                            if _is_tool_result_content_block(sub):
                                _maybe_record(
                                    _extract_inbound_media_url(
                                        cast("Mapping[str, Any]", sub)
                                    )
                                )
                    continue
                _maybe_record(_extract_inbound_media_url(h_item))
        elif isinstance(msg, (ToolMessage, FunctionMessage)):
            if not isinstance(msg.content, list):
                continue
            for t_item in msg.content:
                if _is_tool_result_content_block(t_item):
                    _maybe_record(
                        _extract_inbound_media_url(cast("Mapping[str, Any]", t_item))
                    )
    return found


def _clean_and_validate_b64(raw_b64: Any, *, strict: bool = False) -> str:
    """Return clean base64 ASCII text, rejecting `PathLike` or non-bytes/str inputs."""
    if isinstance(raw_b64, os.PathLike):
        msg_path = (
            "Loading from local files is no longer supported for security reasons. "
            "Pass base64 data, a data URI, a GCS URI, or an HTTP(S) URL instead."
        )
        raise ValueError(msg_path)
    if isinstance(raw_b64, bytes):
        return base64.b64encode(raw_b64).decode("ascii")
    if not isinstance(raw_b64, str):
        msg_type = (
            f"Expected base64 string or bytes for media data, "
            f"got {type(raw_b64).__name__}."
        )
        raise ValueError(msg_type)
    if _is_data_uri(raw_b64):
        stripped_b64 = raw_b64.lstrip()
        match = re.match(
            r"^data:([^;]+);base64,(.*)$",
            stripped_b64,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            msg = f"Invalid base64 data URI format: {stripped_b64[:80]}..."
            raise ValueError(msg)
        raw_b64 = match.group(2)
    cleaned = re.sub(r"\s+", "", raw_b64)
    if strict:
        if not re.match(r"^[A-Za-z0-9+/]*={0,2}$", cleaned) or len(cleaned) % 4 != 0:
            msg = "Data should be valid base64"
            raise ValueError(msg)
        try:
            base64.b64decode(cleaned)
        except Exception as exc:
            msg = "Data should be valid base64"
            raise ValueError(msg) from exc
    return cleaned


def _modality_from_mime(mime_type: str | None, fallback: str = "document") -> str:
    """Classify an Interactions content block type from a MIME type."""
    if not mime_type:
        return fallback
    m = mime_type.lower()
    if m.startswith("image/"):
        return "image"
    if m.startswith("audio/"):
        return "audio"
    if m.startswith("video/"):
        return "video"
    return "document"


def _attach_media_options(
    target: dict[str, Any],
    source_block: Mapping[str, Any],
    *,
    model: str | None,
    default_resolution: Any = None,
) -> dict[str, Any]:
    """Attach `resolution`, `processing`, and warn on unsupported `video_metadata`."""
    raw_extras = source_block.get("extras")
    extras: Mapping[str, Any] = raw_extras if isinstance(raw_extras, Mapping) else {}

    if (
        source_block.get("video_metadata") is not None
        or extras.get("video_metadata") is not None
    ):
        warnings.warn(
            "Gemini Interactions API does not support 'video_metadata'; omitting.",
            UserWarning,
            stacklevel=3,
        )
    block_type = target.get("type")
    if block_type in ("image", "video"):
        raw_res: Any = None
        for cand_r in (
            source_block.get("resolution"),
            source_block.get("media_resolution"),
            extras.get("resolution"),
            extras.get("media_resolution"),
            default_resolution,
        ):
            if cand_r is not None:
                raw_res = cand_r
                break
        res = _normalize_resolution(raw_res, model=model)
        if res:
            target["resolution"] = res
    if block_type == "video":
        raw_proc: Any = None
        for cand_p in (
            source_block.get("processing"),
            source_block.get("media_processing"),
            extras.get("processing"),
            extras.get("media_processing"),
        ):
            if cand_p is not None:
                raw_proc = cand_p
                break
        proc = _normalize_processing(raw_proc)
        if proc is not None:
            target["processing"] = proc
        v_name = source_block.get("name") or extras.get("name")
        if v_name:
            target["name"] = str(v_name)
    if block_type == "audio":
        sr_val = (
            source_block["sample_rate"]
            if source_block.get("sample_rate") is not None
            else extras.get("sample_rate")
        )
        if sr_val is not None:
            target["sample_rate"] = int(sr_val)
        ch_val = (
            source_block["channels"]
            if source_block.get("channels") is not None
            else extras.get("channels")
        )
        if ch_val is not None:
            target["channels"] = int(ch_val)
    return target


def _build_uri_or_fetched_block(
    uri_or_url: str,
    mime_type: str | None,
    preferred_kind: str,
    source_block: Mapping[str, Any],
    *,
    use_vertexai: bool,
    model: str | None,
    default_resolution: Any,
    image_loader: ImageBytesLoader,
    provider_kind: str = "unknown",
    allow_external_fetch: bool = False,
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
) -> dict[str, Any] | None:
    """Build a media/text block from a data URI, GCS URI, Files/YT URI, or HTTP URL."""
    if _is_data_uri(uri_or_url):
        stripped_data = uri_or_url.lstrip()
        match = re.match(
            r"^data:([^;]+);base64,(.*)$",
            stripped_data,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not match:
            msg = f"Invalid base64 data URI format: {stripped_data[:80]}..."
            raise ValueError(msg)
        parsed_mime = (mime_type or match.group(1).strip()).lower()
        b64_data = re.sub(r"\s+", "", match.group(2))
        if parsed_mime == "text/plain":
            decoded = base64.b64decode(b64_data).decode("utf-8", errors="replace")
            return {"type": "text", "text": decoded} if decoded else None
        kind = (
            preferred_kind
            if preferred_kind in ("image", "audio", "video")
            else _modality_from_mime(parsed_mime, preferred_kind)
        )
        if kind == "audio":
            parsed_mime = _normalize_audio_mime(parsed_mime)
        blk: dict[str, Any] = {"type": kind, "data": b64_data, "mime_type": parsed_mime}
        return _attach_media_options(
            blk, source_block, model=model, default_resolution=default_resolution
        )

    if not allow_external_fetch and provider_kind == "foreign":
        logger.warning(
            "Dropping foreign AIMessage media URI without fetching (scheme: %s).",
            urlparse(uri_or_url).scheme.lower() or "non-url",
        )
        return None

    if _is_gs_uri(uri_or_url):
        canon_gs = _canonicalize_uri_scheme(uri_or_url)
        if not use_vertexai:
            msg_gs = (
                "Dropping Google Cloud Storage file URI because the target Gemini "
                "Developer API does not support gs:// references."
            )
            logger.warning(msg_gs)
            warnings.warn(msg_gs, UserWarning, stacklevel=3)
            return None
        guessed = mime_type or mimetypes.guess_type(canon_gs)[0]
        kind_gs = (
            preferred_kind
            if preferred_kind in ("image", "audio", "video")
            else _modality_from_mime(guessed, preferred_kind)
        )
        out_gs: dict[str, Any] = {"type": kind_gs, "uri": canon_gs}
        if guessed:
            out_gs["mime_type"] = (
                _normalize_audio_mime(guessed) if kind_gs == "audio" else guessed
            )
        return _attach_media_options(
            out_gs, source_block, model=model, default_resolution=default_resolution
        )

    if _is_youtube_url(uri_or_url):
        if not allow_external_fetch and mime_type == "text/plain":
            logger.warning(
                "Dropping incompatible text-plain YouTube URL on AIMessage replay."
            )
            return None
        yt_blk: dict[str, Any] = {
            "type": "video",
            "uri": _canonicalize_uri_scheme(uri_or_url),
        }
        if mime_type:
            yt_blk["mime_type"] = mime_type
        return _attach_media_options(
            yt_blk, source_block, model=model, default_resolution=default_resolution
        )

    if _is_provider_uri(uri_or_url) or _is_files_api_id(uri_or_url):
        canon_p = (
            uri_or_url.strip()
            if _is_files_api_id(uri_or_url)
            else _canonicalize_uri_scheme(uri_or_url)
        )
        guessed_p = mime_type or mimetypes.guess_type(canon_p)[0]
        kind_p = (
            preferred_kind
            if preferred_kind in ("image", "audio", "video")
            else _modality_from_mime(guessed_p, preferred_kind)
        )
        out_p: dict[str, Any] = {"type": kind_p, "uri": canon_p}
        if guessed_p:
            out_p["mime_type"] = (
                _normalize_audio_mime(guessed_p) if kind_p == "audio" else guessed_p
            )
        return _attach_media_options(
            out_p, source_block, model=model, default_resolution=default_resolution
        )

    if not allow_external_fetch:
        logger.warning(
            "Dropping external or unsupported media URI on AIMessage replay "
            "without fetching (scheme: %s).",
            urlparse(uri_or_url).scheme.lower() or "non-url",
        )
        return None

    b64_fetched, detected_mime = _fetch_external_url(
        uri_or_url, image_loader, prefetched_urls=prefetched_urls
    )
    eff_mime = mime_type or detected_mime
    if eff_mime == "text/plain":
        decoded_txt = base64.b64decode(b64_fetched).decode("utf-8", errors="replace")
        return {"type": "text", "text": decoded_txt} if decoded_txt else None
    kind_f = (
        preferred_kind
        if preferred_kind in ("image", "audio", "video")
        else _modality_from_mime(eff_mime, preferred_kind)
    )
    if kind_f == "audio":
        eff_mime = _normalize_audio_mime(eff_mime)
    elif not eff_mime:
        defaults = {
            "image": "image/jpeg",
            "video": "video/mp4",
            "document": "application/pdf",
        }
        eff_mime = defaults.get(kind_f, "application/pdf")
    out_f: dict[str, Any] = {"type": kind_f, "data": b64_fetched, "mime_type": eff_mime}
    return _attach_media_options(
        out_f, source_block, model=model, default_resolution=default_resolution
    )


def _convert_block_to_interaction_content(
    part: Any,
    *,
    use_vertexai: bool,
    model: str | None,
    default_resolution: Any,
    image_loader: ImageBytesLoader,
    provider_kind: str = "unknown",
    allow_external_fetch: bool = False,
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
) -> dict[str, Any] | None:
    """Convert a single content block (v1/v0/OAI/Anthropic/native) to `ContentParam`."""
    if isinstance(part, str):
        return {"type": "text", "text": part} if part else None
    if not isinstance(part, Mapping):
        msg = f"Unsupported message content block type: {type(part)}"
        raise ChatGoogleGenerativeAIError(msg)

    if set(part.keys()) == {"cachePoint"} or part.get("type") in (
        "cache_point",
        "refusal",
    ):
        return None

    btype = part.get("type")
    if btype == "non_standard":
        inner = part.get("value")
        if provider_kind != "foreign" and isinstance(inner, Mapping):
            return _convert_block_to_interaction_content(
                inner,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )
        return None

    if btype is None:
        if "text" in part and isinstance(part["text"], str):
            return {"type": "text", "text": part["text"]} if part["text"] else None
        rendered = str(dict(part))
        return {"type": "text", "text": rendered} if rendered else None

    if btype == "text":
        raw_txt = part.get("text")
        txt_str = "" if raw_txt is None else str(raw_txt)
        if not txt_str:
            return None
        txt_blk: dict[str, Any] = {"type": "text", "text": txt_str}
        raw_ann = part.get("annotations")
        if isinstance(raw_ann, list):
            speech_anns = [
                copy.deepcopy(dict(a))
                for a in raw_ann
                if isinstance(a, Mapping) and a.get("type") == "speech_metadata"
            ]
            if speech_anns:
                txt_blk["annotations"] = speech_anns
        return txt_blk

    if btype == "text-plain":
        if "text" in part and isinstance(part["text"], str):
            return {"type": "text", "text": part["text"]} if part["text"] else None
        if b64 := (part.get("base64") or part.get("data")):
            raw = base64.b64decode(_clean_and_validate_b64(b64))
            decoded_tp = raw.decode("utf-8", errors="replace")
            return {"type": "text", "text": decoded_tp} if decoded_tp else None
        if url_tp := part.get("url"):
            return _build_uri_or_fetched_block(
                str(url_tp),
                "text/plain",
                "document",
                part,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )
        return None

    if btype == "image_url":
        raw_iu = part.get("image_url")
        if isinstance(raw_iu, Mapping):
            if "url" not in raw_iu:
                msg = f"Unrecognized image_url block: {raw_iu}"
                raise ValueError(msg)
            url_str = str(raw_iu["url"])
        elif isinstance(raw_iu, str):
            url_str = raw_iu
        else:
            msg = f"Unrecognized image_url block: {part}"
            raise ValueError(msg)
        return _build_uri_or_fetched_block(
            url_str,
            None,
            "image",
            part,
            use_vertexai=use_vertexai,
            model=model,
            default_resolution=default_resolution,
            image_loader=image_loader,
            provider_kind=provider_kind,
            allow_external_fetch=allow_external_fetch,
            prefetched_urls=prefetched_urls,
        )

    if btype == "input_audio":
        ia = part.get("input_audio")
        if not isinstance(ia, Mapping) or "data" not in ia:
            msg = f"Invalid input_audio block: {part}"
            raise ValueError(msg)
        fmt_mime = _normalize_audio_mime(ia.get("format"))
        aud_blk: dict[str, Any] = {
            "type": "audio",
            "data": _clean_and_validate_b64(ia["data"]),
            "mime_type": fmt_mime,
        }
        return _attach_media_options(
            aud_blk, part, model=model, default_resolution=default_resolution
        )

    # Anthropic image/document blocks with nested `source`
    if btype in ("image", "document") and isinstance(part.get("source"), Mapping):
        src = cast("Mapping[str, Any]", part["source"])
        src_type = src.get("type")
        src_mime = src.get("media_type") or part.get("mime_type")
        if src_type == "base64" and src.get("data"):
            out_a: dict[str, Any] = {
                "type": btype,
                "data": _clean_and_validate_b64(src["data"]),
                "mime_type": src_mime
                or ("image/jpeg" if btype == "image" else "application/pdf"),
            }
            return _attach_media_options(
                out_a, part, model=model, default_resolution=default_resolution
            )
        if src_type == "text" and isinstance(src.get("data"), str):
            return {"type": "text", "text": src["data"]} if src["data"] else None
        if src_type == "url" and src.get("url"):
            return _build_uri_or_fetched_block(
                str(src["url"]),
                src_mime,
                str(btype),
                part,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )

    # OpenAI file block `{"type": "file", "file": {"filename": ..., "file_data": ...}}`
    if btype == "file" and isinstance(part.get("file"), Mapping):
        fobj = cast("Mapping[str, Any]", part["file"])
        fname = fobj.get("filename") or ""
        guessed_fmime = fobj.get("mime_type") or (
            mimetypes.guess_type(str(fname))[0] if fname else None
        )
        fdata = fobj.get("file_data")
        if isinstance(fdata, str) and fdata:
            if "://" in fdata or _is_data_uri(fdata) or _is_files_api_id(fdata):
                return _build_uri_or_fetched_block(
                    fdata,
                    guessed_fmime,
                    "document",
                    part,
                    use_vertexai=use_vertexai,
                    model=model,
                    default_resolution=default_resolution,
                    image_loader=image_loader,
                    provider_kind=provider_kind,
                    allow_external_fetch=allow_external_fetch,
                    prefetched_urls=prefetched_urls,
                )
            eff_m = guessed_fmime or "application/pdf"
            kind_oai = _modality_from_mime(eff_m, "document")
            return {
                "type": kind_oai,
                "data": _clean_and_validate_b64(fdata),
                "mime_type": _normalize_audio_mime(eff_m)
                if kind_oai == "audio"
                else eff_m,
            }
        if fobj.get("file_id") and provider_kind != "foreign":
            return _build_uri_or_fetched_block(
                str(fobj["file_id"]),
                guessed_fmime,
                "document",
                part,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )
        return None

    if btype in ("media", "file_data"):
        m_mime = part.get("mime_type")
        if "data" in part and part["data"] is not None:
            if not m_mime:
                msg = f"Missing mime_type in media part: {part}"
                raise ValueError(msg)
            b64_m = _clean_and_validate_b64(part["data"], strict=True)
            if m_mime == "text/plain":
                raw_t = base64.b64decode(b64_m).decode("utf-8", errors="replace")
                return {"type": "text", "text": raw_t} if raw_t else None
            kind_m = _modality_from_mime(m_mime, "document")
            blk_m: dict[str, Any] = {
                "type": kind_m,
                "data": b64_m,
                "mime_type": _normalize_audio_mime(m_mime)
                if kind_m == "audio"
                else m_mime,
            }
            return _attach_media_options(
                blk_m, part, model=model, default_resolution=default_resolution
            )
        if f_uri := part.get("file_uri"):
            return _build_uri_or_fetched_block(
                str(f_uri),
                m_mime,
                _modality_from_mime(m_mime, "document"),
                part,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )
        msg = f"Media part must have either data or file_uri: {part}"
        raise ValueError(msg)

    if btype in ("image", "audio", "video", "document", "file"):
        raw_mime = part.get("mime_type")
        pref_kind = (
            _modality_from_mime(raw_mime, "document")
            if btype in ("file", "document")
            else str(btype)
        )

        # Native `uri` or v1 `file_id` or `url` / v0 `source_type == "url"`
        uri_candidate = part.get("uri") or (
            part.get("file_id") if provider_kind != "foreign" else None
        )
        if not uri_candidate and (part.get("source_type") == "url" or "url" in part):
            url_val = part.get("url")
            if url_val:
                uri_candidate = url_val

        # Inline base64 (`base64` or `data`)
        inline_raw = part.get("base64")
        if inline_raw is None and part.get("source_type") in (None, "base64"):
            inline_raw = part.get("data")

        if uri_candidate:
            built_uri = _build_uri_or_fetched_block(
                str(uri_candidate),
                raw_mime,
                pref_kind,
                part,
                use_vertexai=use_vertexai,
                model=model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=allow_external_fetch,
                prefetched_urls=prefetched_urls,
            )
            if built_uri is not None or inline_raw is None:
                return built_uri
        if inline_raw is not None:
            b64_val = _clean_and_validate_b64(inline_raw)
            eff_mime_b = raw_mime
            if not eff_mime_b:
                try:
                    kind_guess = filetype.guess(base64.b64decode(b64_val))
                    if kind_guess:
                        eff_mime_b = kind_guess.mime
                except Exception:
                    eff_mime_b = None
            if eff_mime_b == "text/plain":
                txt_dec = base64.b64decode(b64_val).decode("utf-8", errors="replace")
                return {"type": "text", "text": txt_dec} if txt_dec else None
            actual_kind = (
                _modality_from_mime(eff_mime_b, "document")
                if btype in ("file", "document")
                else pref_kind
            )
            if actual_kind == "audio":
                eff_mime_b = _normalize_audio_mime(eff_mime_b)
            elif not eff_mime_b:
                defaults_b = {
                    "image": "image/jpeg",
                    "video": "video/mp4",
                    "document": "application/pdf",
                }
                eff_mime_b = defaults_b.get(actual_kind, "application/pdf")
            blk_in: dict[str, Any] = {
                "type": actual_kind,
                "data": b64_val,
                "mime_type": eff_mime_b,
            }
            return _attach_media_options(
                blk_in, part, model=model, default_resolution=default_resolution
            )

        if part.get("file_id") and provider_kind == "foreign":
            return None
        msg = (
            f"Data content block must contain 'url', 'uri', 'base64', or 'data': {part}"
        )
        raise ValueError(msg)

    msg = f"Unrecognized message part type: {btype}."
    raise ValueError(msg)


def _normalize_mcp_result_payload(output_val: Any) -> Any:
    """Normalize an MCP server tool result for Speakeasy `MCPServerToolResultStep`.

    Keeps `output_val` as a `list` only when it is non-empty and every element is a
    valid `FunctionResultSubcontent` (`{"type": "text", "text": str}` or
    `{"type": "image", "data" | "uri": str}`). Any other list (empty lists, JSON
    object arrays, primitive lists, or non-`text`/`image` blocks) is serialized with
    `json.dumps(..., ensure_ascii=False, default=str)` so stateless multi-turn replay
    passes Speakeasy union validation cleanly.
    """
    if isinstance(output_val, str):
        return output_val
    if isinstance(output_val, Mapping):
        return copy.deepcopy(dict(output_val))
    if isinstance(output_val, list):
        if output_val:
            subcontents: list[dict[str, Any]] = []
            valid = True
            for item in output_val:
                if not isinstance(item, Mapping):
                    valid = False
                    break
                itype = item.get("type")
                if itype == "text":
                    txt = item.get("text")
                    ann = item.get("annotations")
                    if not isinstance(txt, str) or (
                        ann is not None and not isinstance(ann, list)
                    ):
                        valid = False
                        break
                    if txt:
                        text_entry: dict[str, Any] = {"type": "text", "text": txt}
                        if isinstance(ann, list):
                            text_entry["annotations"] = copy.deepcopy(ann)
                        subcontents.append(text_entry)
                elif itype == "image":
                    raw_data = item.get("data")
                    raw_uri = item.get("uri")
                    if raw_data is not None and not isinstance(raw_data, str):
                        valid = False
                        break
                    if raw_uri is not None and not isinstance(raw_uri, str):
                        valid = False
                        break
                    clean_data: str | None = None
                    if isinstance(raw_data, str) and raw_data:
                        try:
                            clean_data = _clean_and_validate_b64(raw_data)
                        except ValueError:
                            valid = False
                            break
                    clean_uri = (
                        raw_uri if isinstance(raw_uri, str) and raw_uri else None
                    )
                    if clean_data is None and clean_uri is None:
                        valid = False
                        break
                    img_entry: dict[str, Any] = {"type": "image"}
                    if clean_data is not None:
                        img_entry["data"] = clean_data
                    if clean_uri is not None:
                        img_entry["uri"] = clean_uri
                    mime = item.get("mime_type")
                    if isinstance(mime, str) and mime:
                        img_entry["mime_type"] = mime
                    res = _unwrap_enum_value(item.get("resolution"))
                    if isinstance(res, str) and res:
                        norm_res = res.lower().removeprefix("media_resolution_")
                        if norm_res and norm_res != "unspecified":
                            img_entry["resolution"] = norm_res
                    subcontents.append(img_entry)
                else:
                    valid = False
                    break
            if valid:
                return subcontents or ""
        return json.dumps(output_val, ensure_ascii=False, default=str)
    if output_val is None:
        return ""
    if isinstance(output_val, (bool, int, float, tuple)):
        return json.dumps(output_val, ensure_ascii=False, default=str)
    return str(output_val)


def _server_block_to_step(
    block: Mapping[str, Any],
    *,
    last_call_ids: dict[str, str],
    fallback_id: str,
    provider_kind: _ModelProviderKind = "native",
) -> dict[str, Any] | None:
    """Convert a native/v0/v1 server tool block into an Interactions `StepParam`."""
    btype = block.get("type")
    if btype in _RAW_SERVER_STEP_TYPES and not (
        btype == "code_execution_result" and "code_execution_result" in block
    ):
        copied = copy.deepcopy(dict(block))
        copied.pop("index", None)
        if btype == "mcp_server_tool_result":
            copied["result"] = _normalize_mcp_result_payload(copied.get("result"))
        if "signature" in copied:
            raw_copied_sig = copied.pop("signature")
            if provider_kind != "foreign":
                norm_s = _normalize_signature_b64(raw_copied_sig)
                if norm_s:
                    copied["signature"] = norm_s
        return copied

    raw_extras = block.get("extras")
    extras: Mapping[str, Any] = raw_extras if isinstance(raw_extras, Mapping) else {}
    sig = _extract_trusted_signature(block, provider_kind=provider_kind)

    if btype == "executable_code":
        cid = str(block.get("id") or fallback_id)
        last_call_ids["code_execution"] = cid
        step: dict[str, Any] = {
            "type": "code_execution_call",
            "id": cid,
            "arguments": {
                "code": str(block.get("executable_code", "")),
                "language": "python",
            },
        }
        if sig:
            step["signature"] = sig
        return step

    if btype == "code_execution_result":
        cid = str(
            block.get("tool_call_id")
            or last_call_ids.get("code_execution")
            or fallback_id
        )
        outcome = block.get("outcome", 1)
        res_step: dict[str, Any] = {
            "type": "code_execution_result",
            "call_id": cid,
            "result": str(block.get("code_execution_result", "")),
        }
        if outcome not in (None, 1, "OUTCOME_OK", "success"):
            res_step["is_error"] = True
        if sig:
            res_step["signature"] = sig
        return res_step

    if btype == "server_tool_call":
        name = str(block.get("name") or "")
        step_type = str(extras.get("step_type") or "")
        if name == _MEDIA_PROCESSING_TOOL_NAME or step_type == "processing_call":
            return None
        cid = str(block.get("id") or fallback_id)
        raw_b_args = block.get("args")
        args: Mapping[str, Any] = raw_b_args if isinstance(raw_b_args, Mapping) else {}
        if name == "code_interpreter" or step_type == "code_execution_call":
            last_call_ids["code_execution"] = cid
            out: dict[str, Any] = {
                "type": "code_execution_call",
                "id": cid,
                "arguments": {
                    "code": str(args.get("code", "")),
                    "language": "python",
                },
            }
        elif name == "google_search" or step_type == "google_search_call":
            last_call_ids["google_search"] = cid
            out = {
                "type": "google_search_call",
                "id": cid,
                "arguments": copy.deepcopy(dict(args)),
            }
            if extras.get("search_type"):
                out["search_type"] = extras["search_type"]
        elif name == "url_context" or step_type == "url_context_call":
            last_call_ids["url_context"] = cid
            out = {
                "type": "url_context_call",
                "id": cid,
                "arguments": copy.deepcopy(dict(args)),
            }
        elif name == "google_maps" or step_type == "google_maps_call":
            last_call_ids["google_maps"] = cid
            out = {"type": "google_maps_call", "id": cid}
            if args:
                out["arguments"] = copy.deepcopy(dict(args))
        elif name == "file_search" or step_type == "file_search_call":
            last_call_ids["file_search"] = cid
            out = {"type": "file_search_call", "id": cid}
        elif step_type == "mcp_server_tool_call" or extras.get("server_name"):
            last_call_ids["mcp_server"] = cid
            mcp_out: dict[str, Any] = {
                "type": "mcp_server_tool_call",
                "id": cid,
                "name": name,
                "server_name": str(extras.get("server_name", "")),
                "arguments": copy.deepcopy(dict(args)),
            }
            return mcp_out
        elif name == "retrieval" or step_type == "retrieval_call":
            last_call_ids["retrieval"] = cid
            out = {
                "type": "retrieval_call",
                "id": cid,
                "arguments": copy.deepcopy(dict(args)),
            }
            if extras.get("retrieval_type"):
                out["retrieval_type"] = extras["retrieval_type"]
        else:
            return None
        if sig:
            out["signature"] = sig
        return out

    if btype == "server_tool_result":
        block_tag = str(extras.get("step_type") or extras.get("block_type") or "")
        if block_tag in (_MEDIA_PROCESSING_TOOL_NAME, "processing_result"):
            return None
        is_err = block.get("status") == "error"
        output_val = block.get("output")
        if block_tag in ("code_execution_result", ""):
            cid = str(
                block.get("tool_call_id")
                or last_call_ids.get("code_execution")
                or fallback_id
            )
            ce_out: dict[str, Any] = {
                "type": "code_execution_result",
                "call_id": cid,
                "result": str(output_val if output_val is not None else ""),
            }
            if is_err or extras.get("outcome") == 2:
                ce_out["is_error"] = True
            if sig:
                ce_out["signature"] = sig
            return ce_out
        if block_tag in (
            "google_search_result",
            "url_context_result",
            "google_maps_result",
        ):
            key = block_tag.removesuffix("_result")
            cid = str(
                block.get("tool_call_id") or last_call_ids.get(key) or fallback_id
            )
            sr_out: dict[str, Any] = {
                "type": block_tag,
                "call_id": cid,
                "result": copy.deepcopy(output_val)
                if isinstance(output_val, list)
                else [],
            }
            if is_err and block_tag != "google_maps_result":
                sr_out["is_error"] = True
            if sig:
                sr_out["signature"] = sig
            return sr_out
        if block_tag in ("file_search_result", "retrieval_result"):
            key = block_tag.removesuffix("_result")
            cid = str(
                block.get("tool_call_id") or last_call_ids.get(key) or fallback_id
            )
            fr_out: dict[str, Any] = {"type": block_tag, "call_id": cid}
            if is_err and block_tag == "retrieval_result":
                fr_out["is_error"] = True
            if sig:
                fr_out["signature"] = sig
            return fr_out
        if block_tag == "mcp_server_tool_result":
            cid = str(
                block.get("tool_call_id")
                or last_call_ids.get("mcp_server")
                or fallback_id
            )
            mcp_res: dict[str, Any] = {
                "type": "mcp_server_tool_result",
                "call_id": cid,
                "result": _normalize_mcp_result_payload(output_val),
            }
            if extras.get("name"):
                mcp_res["name"] = str(extras["name"])
            if extras.get("server_name"):
                mcp_res["server_name"] = str(extras["server_name"])
            return mcp_res
    return None


def _extract_reasoning_text(block: Mapping[str, Any]) -> str:
    """Extract summary text from v0 `thinking` or v1 `reasoning` block."""
    for key in ("thinking", "reasoning", "reasoning_content", "text"):
        val = block.get(key)
        if isinstance(val, str) and val:
            return val
    summary = block.get("summary")
    if isinstance(summary, list):
        parts = [
            s.strip()
            for item in summary
            if isinstance(item, Mapping)
            and isinstance((s := item.get("text")), str)
            and s.strip()
        ]
        return "\n".join(parts)
    return ""


def _build_thought_step(sig: str, summary_text: str = "") -> dict[str, Any]:
    step: dict[str, Any] = {"type": "thought", "signature": sig}
    if summary_text:
        step["summary"] = [{"type": "text", "text": summary_text}]
    return step


def _convert_tool_result_payload(
    raw_content: Any,
    *,
    call_id: str,
    name: str | None,
    is_error: bool,
    use_vertexai: bool,
    model: str | None,
    default_resolution: Any,
    image_loader: ImageBytesLoader,
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build `(FunctionResultStepParam, trailing_followup_media_blocks)`.

    String tool payloads stay plain strings (never parsed into dicts via `json.loads`)
    and `Mapping` payloads are JSON-serialized consistently so empty lists `[]`, empty
    dicts `{}`, `null`, or mixed-type arrays never trigger Vertex AI stateful HTTP 400
    protobuf `Struct` value validation errors.
    """
    step: dict[str, Any] = {"type": "function_result", "call_id": call_id}
    if name:
        step["name"] = name
    if is_error:
        step["is_error"] = True

    if isinstance(raw_content, str):
        step["result"] = raw_content
        return step, []

    if isinstance(raw_content, Mapping):
        step["result"] = json.dumps(dict(raw_content), ensure_ascii=False, default=str)
        return step, []

    if isinstance(raw_content, list):
        subcontents: list[dict[str, Any]] = []
        followup_media: list[dict[str, Any]] = []
        for item in raw_content:
            if _is_tool_result_content_block(item):
                converted = _convert_block_to_interaction_content(
                    item,
                    use_vertexai=use_vertexai,
                    model=model,
                    default_resolution=default_resolution,
                    image_loader=image_loader,
                    allow_external_fetch=True,
                    prefetched_urls=prefetched_urls,
                )
                if converted is None:
                    continue
                if converted["type"] in ("text", "image"):
                    subcontents.append(converted)
                else:
                    followup_media.append(converted)
            elif isinstance(item, str):
                if item:
                    subcontents.append({"type": "text", "text": item})
            else:
                plain_item = dict(item) if isinstance(item, Mapping) else item
                subcontents.append(
                    {
                        "type": "text",
                        "text": json.dumps(plain_item, ensure_ascii=False, default=str),
                    }
                )

        if followup_media:
            step["result"] = subcontents or "See attached media."
            return step, followup_media
        step["result"] = subcontents or ""
        return step, []

    step["result"] = str(raw_content) if raw_content is not None else ""
    return step, []


def _extract_system_text(
    messages: Sequence[BaseMessage],
) -> str | None:
    """Join all `SystemMessage` text blocks with `'\\n\\n'`."""
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, SystemMessage):
            continue
        if isinstance(msg.content, str):
            if msg.content:
                chunks.append(msg.content)
        elif isinstance(msg.content, list):
            block_texts: list[str] = []
            for blk in msg.content:
                if isinstance(blk, str) and blk:
                    block_texts.append(blk)
                elif (
                    isinstance(blk, Mapping)
                    and blk.get("type") in ("text", "text-plain", None)
                    and isinstance(blk.get("text"), str)
                    and blk["text"]
                ):
                    block_texts.append(blk["text"])
            if block_texts:
                chunks.append("".join(block_texts))
    return "\n\n".join(chunks) if chunks else None


def _is_empty_human_text_block(block: Mapping[str, Any]) -> bool:
    """Return True when `block` is a plain text mapping carrying only empty text."""
    btype = block.get("type")
    if btype == "text":
        raw_txt = block.get("text")
        return raw_txt is None or str(raw_txt) == ""
    if btype == "text-plain":
        if block.get("base64") or block.get("data") or block.get("url"):
            return False
        raw_tp = block.get("text")
        return raw_tp is None or (isinstance(raw_tp, str) and raw_tp == "")
    if btype is None and "text" in block and isinstance(block["text"], str):
        return block["text"] == ""
    return False


def _is_real_user_prompt(msg: BaseMessage) -> bool:
    """Return True when a `HumanMessage` starts a user turn (not only `tool_result`)."""
    if not isinstance(msg, HumanMessage) or not msg.content:
        return False
    if isinstance(msg.content, str):
        return True
    return any(
        (isinstance(b, str) and bool(b))
        or (
            isinstance(b, Mapping)
            and b.get("type") not in ("tool_result", "cache_point", "refusal")
            and set(b.keys()) != {"cachePoint"}
            and not _is_empty_human_text_block(b)
        )
        for b in msg.content
    )


def _convert_ai_message_to_steps(
    msg: AIMessage,
    *,
    msg_idx: int,
    in_active_loop: bool,
    effective_model: str,
    use_vertexai: bool,
    default_resolution: Any,
    image_loader: ImageBytesLoader,
    known_call_names: dict[str, str],
    pending_unided_calls: list[tuple[str, str]],
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
) -> list[dict[str, Any]]:
    """Convert an `AIMessage` into strictly ordered Interactions steps per segment."""
    provider_kind = _classify_model_provider(
        msg.response_metadata.get("model_provider")
    )
    is_foreign = provider_kind == "foreign"

    # Build authoritative list of tool calls (from msg.tool_calls, fallback to
    # legacy additional_kwargs["function_call"] or Anthropic tool_use blocks).
    normalized_tcs: list[dict[str, Any]] = []
    raw_tcs = list(msg.tool_calls or [])
    if not raw_tcs and isinstance(msg.additional_kwargs.get("function_call"), Mapping):
        fc_legacy = msg.additional_kwargs["function_call"]
        raw_fc_args = fc_legacy.get("arguments", "{}")
        try:
            parsed_fc_args = (
                json.loads(raw_fc_args)
                if isinstance(raw_fc_args, str)
                else dict(raw_fc_args or {})
            )
        except Exception:
            parsed_fc_args = {}
        raw_tcs.append(
            {
                "name": str(fc_legacy.get("name", "")),
                "args": parsed_fc_args if isinstance(parsed_fc_args, dict) else {},
                "id": fc_legacy.get("id"),
            }
        )
    if not raw_tcs and isinstance(msg.content, list):
        for blk in msg.content:
            if isinstance(blk, Mapping) and blk.get("type") == "tool_use":
                inp = blk.get("input")
                raw_tcs.append(
                    {
                        "name": str(blk.get("name", "")),
                        "args": copy.deepcopy(dict(inp))
                        if isinstance(inp, Mapping)
                        else {},
                        "id": blk.get("id"),
                    }
                )

    for tc_idx, tc in enumerate(raw_tcs):
        raw_id = tc.get("id")
        tc_name = str(tc.get("name", ""))
        if isinstance(raw_id, str) and raw_id:
            det_id = raw_id
        else:
            det_id = f"call_{msg_idx}_{tc_idx}_{tc_name or 'fn'}"
            pending_unided_calls.append((tc_name, det_id))
        if tc_name:
            known_call_names[det_id] = tc_name
        raw_args = tc.get("args", {})
        if isinstance(raw_args, str):
            try:
                loaded_args = json.loads(raw_args)
                args_dict = loaded_args if isinstance(loaded_args, dict) else {}
            except Exception:
                args_dict = {}
        elif isinstance(raw_args, Mapping):
            args_dict = copy.deepcopy(dict(raw_args))
        else:
            args_dict = {}
        tc_entry: dict[str, Any] = {
            "id": det_id,
            "orig_id": raw_id,
            "name": tc_name,
            "arguments": args_dict,
        }
        tc_sig = _extract_trusted_signature(tc, provider_kind=provider_kind)
        if tc_sig:
            tc_entry["sig"] = tc_sig
        normalized_tcs.append(tc_entry)

    fc_sigs_map: dict[Any, Any] = {}
    indexed_fc_sigs: dict[int, Any] = {}
    if not is_foreign:
        raw_map = msg.additional_kwargs.get(
            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
        )
        if isinstance(raw_map, Mapping):
            fc_sigs_map = dict(raw_map)
        indexed_fc_sigs = _function_call_signatures_from_content(msg)

    for tc_idx, tc_entry in enumerate(normalized_tcs):
        if "sig" not in tc_entry:
            cand = (
                fc_sigs_map.get(tc_entry["id"])
                or (
                    fc_sigs_map.get(tc_entry["orig_id"])
                    if tc_entry["orig_id"]
                    else None
                )
                or indexed_fc_sigs.get(tc_idx)
            )
            norm_cand = _normalize_signature_b64(cand)
            if norm_cand:
                tc_entry["sig"] = norm_cand

    raw_items: list[Any]
    if isinstance(msg.content, str):
        raw_items = [msg.content] if msg.content else []
    elif isinstance(msg.content, list):
        raw_items = list(msg.content)
    else:
        raw_items = []

    v0_aud = msg.additional_kwargs.get("audio")
    if (
        not is_foreign
        and isinstance(v0_aud, bytes)
        and v0_aud
        and not any(
            isinstance(it, Mapping) and it.get("type") == "audio" for it in raw_items
        )
    ):
        raw_items.append(
            {
                "type": "audio",
                "base64": base64.b64encode(v0_aud).decode("ascii"),
                "mime_type": "audio/wav",
            }
        )

    # Pre-extract signatures from inline/appended tool_call blocks (critical when an
    # AIMessage was persisted/restored from v1 content_blocks without additional_kwargs)
    tc_block_types = ("tool_call", "tool_use", "function_call")
    prescan_cursor = 0
    prescan_seen: set[int] = set()
    saw_tc_block = False
    has_interleaved_tc = False
    for raw_it in raw_items:
        if isinstance(raw_it, str):
            if raw_it and saw_tc_block:
                has_interleaved_tc = True
            continue
        if not isinstance(raw_it, Mapping):
            continue
        it_type = raw_it.get("type")
        if it_type == "non_standard" and isinstance(raw_it.get("value"), Mapping):
            raw_it = cast("Mapping[str, Any]", raw_it["value"])
            it_type = raw_it.get("type")
        if it_type in tc_block_types:
            saw_tc_block = True
            p_idx: int | None = None
            b_id = raw_it.get("id")
            if b_id:
                for ci, ctc in enumerate(normalized_tcs):
                    if ci not in prescan_seen and ctc["orig_id"] == b_id:
                        p_idx = ci
                        break
            if p_idx is None:
                while prescan_cursor < len(normalized_tcs):
                    if prescan_cursor not in prescan_seen:
                        p_idx = prescan_cursor
                        prescan_cursor += 1
                        break
                    prescan_cursor += 1
            if p_idx is not None:
                prescan_seen.add(p_idx)
                if "sig" not in normalized_tcs[p_idx]:
                    p_sig = _extract_trusted_signature(
                        raw_it, provider_kind=provider_kind
                    )
                    if p_sig:
                        normalized_tcs[p_idx]["sig"] = p_sig
        elif (
            saw_tc_block
            and not _is_rebuilt_tool_call_block(raw_it)
            and it_type
            not in (
                "refusal",
                "cache_point",
            )
        ):
            has_interleaved_tc = True

    steps_out: list[dict[str, Any]] = []
    seg_thoughts: list[dict[str, Any]] = []
    seg_carrier_sigs: list[str] = []
    seg_server_steps: list[dict[str, Any]] = []
    seg_contents: list[dict[str, Any]] = []
    seg_fcs: list[dict[str, Any]] = []
    emitted_thought_sigs: set[str] = set()
    last_call_ids: dict[str, str] = {}
    consumed_tc_indices: set[int] = set()

    def _seg_thought_sigs() -> set[str]:
        return {t["signature"] for t in seg_thoughts if "signature" in t} | set(
            seg_carrier_sigs
        )

    def _flush_segment() -> None:
        has_substantive_step = bool(seg_server_steps or seg_contents or seg_fcs)
        if seg_carrier_sigs:
            if has_substantive_step:
                for c_sig in seg_carrier_sigs:
                    if c_sig not in emitted_thought_sigs and not any(
                        t.get("signature") == c_sig for t in seg_thoughts
                    ):
                        seg_thoughts.append(_build_thought_step(c_sig))
            seg_carrier_sigs.clear()
        if seg_thoughts:
            for t in seg_thoughts:
                if "signature" in t:
                    emitted_thought_sigs.add(t["signature"])
            steps_out.extend(seg_thoughts)
            seg_thoughts.clear()
        if seg_server_steps:
            steps_out.extend(seg_server_steps)
            seg_server_steps.clear()
        if seg_contents:
            steps_out.append({"type": "model_output", "content": list(seg_contents)})
            seg_contents.clear()
        if seg_fcs:
            steps_out.extend(seg_fcs)
            seg_fcs.clear()

    def _append_fc(tc_idx: int) -> None:
        consumed_tc_indices.add(tc_idx)
        tc_e = normalized_tcs[tc_idx]
        tc_sig = tc_e.get("sig")
        if seg_server_steps:
            # Never mix a post-server-tool function_call (or its thought signature)
            # back into an open server-tool segment.
            _flush_segment()
        if (
            tc_sig
            and tc_sig not in _seg_thought_sigs()
            and tc_sig not in emitted_thought_sigs
        ):
            if seg_fcs:
                _flush_segment()
            elif (seg_thoughts or seg_carrier_sigs) and seg_contents:
                # An earlier signed thought signed the model_output; flush it so
                # this new signature sits directly before its function_call.
                _flush_segment()
            elif seg_carrier_sigs and not seg_contents:
                seg_carrier_sigs.clear()
            seg_thoughts.append(_build_thought_step(tc_sig))
        seg_fcs.append(
            {
                "type": "function_call",
                "id": tc_e["id"],
                "name": tc_e["name"],
                "arguments": tc_e["arguments"],
            }
        )

    def _flush_prior_tcs_for_new_sig(next_sig: str) -> None:
        """Emit earlier segments' tool calls before opening `next_sig` in v0/v1."""
        if has_interleaved_tc or not normalized_tcs:
            return
        target_pos: int | None = None
        for ci in range(len(normalized_tcs)):
            if (
                ci not in consumed_tc_indices
                and normalized_tcs[ci].get("sig") == next_sig
            ):
                target_pos = ci
                break
        if target_pos is not None:
            flushed_any = False
            for prior_i in range(target_pos):
                if prior_i not in consumed_tc_indices:
                    _append_fc(prior_i)
                    flushed_any = True
            if flushed_any or seg_fcs:
                _flush_segment()
            return
        cur_sigs = _seg_thought_sigs()
        if cur_sigs and next_sig not in cur_sigs:
            flushed_cur = False
            for ci in range(len(normalized_tcs)):
                if ci in consumed_tc_indices:
                    continue
                if normalized_tcs[ci].get("sig") in cur_sigs:
                    _append_fc(ci)
                    flushed_cur = True
                else:
                    break
            if flushed_cur or seg_fcs:
                _flush_segment()

    inline_cursor = 0
    for item_idx, item in enumerate(raw_items):
        if isinstance(item, str):
            if not item:
                continue
            if seg_fcs or seg_server_steps:
                _flush_segment()
            seg_contents.append({"type": "text", "text": item})
            continue

        if not isinstance(item, Mapping):
            continue

        unwrapped: Mapping[str, Any] = item
        if item.get("type") == "non_standard":
            inner = item.get("value")
            if is_foreign or not isinstance(inner, Mapping):
                continue
            unwrapped = inner

        btype = unwrapped.get("type")
        if btype in _REASONING_BLOCK_TYPES:
            if is_foreign:
                continue
            r_sig = _extract_trusted_signature(unwrapped, provider_kind=provider_kind)
            if not r_sig:
                # Unsigned or unverified non-native thought steps cause HTTP 400
                continue
            r_text = _extract_reasoning_text(unwrapped)
            t_step = _build_thought_step(r_sig, r_text)
            _flush_prior_tcs_for_new_sig(r_sig)
            if (
                seg_fcs
                or seg_server_steps
                or (
                    seg_contents
                    and (seg_thoughts or seg_carrier_sigs)
                    and r_sig not in _seg_thought_sigs()
                )
            ):
                _flush_segment()
            elif (
                seg_carrier_sigs and not seg_contents and r_sig not in seg_carrier_sigs
            ):
                seg_carrier_sigs.clear()
            if r_sig in seg_carrier_sigs:
                seg_carrier_sigs.remove(r_sig)
            seg_thoughts.append(t_step)
            continue

        if btype in ("tool_call", "tool_use", "function_call"):
            # Match by ID or sequential cursor against normalized_tcs
            blk_id = unwrapped.get("id")
            matched_idx: int | None = None
            if blk_id:
                for cand_i, cand_tc in enumerate(normalized_tcs):
                    if (
                        cand_i not in consumed_tc_indices
                        and cand_tc["orig_id"] == blk_id
                    ):
                        matched_idx = cand_i
                        break
            if matched_idx is None:
                while inline_cursor < len(normalized_tcs):
                    if inline_cursor not in consumed_tc_indices:
                        matched_idx = inline_cursor
                        inline_cursor += 1
                        break
                    inline_cursor += 1
            if matched_idx is not None:
                if "sig" not in normalized_tcs[matched_idx]:
                    blk_sig = _extract_trusted_signature(
                        unwrapped, provider_kind=provider_kind
                    )
                    if blk_sig:
                        normalized_tcs[matched_idx]["sig"] = blk_sig
                _append_fc(matched_idx)
            continue

        if _is_rebuilt_tool_call_block(unwrapped):
            continue

        if btype in _RAW_SERVER_STEP_TYPES or btype in (
            "executable_code",
            "code_execution_result",
            "server_tool_call",
            "server_tool_result",
        ):
            if (
                is_foreign
                and btype == "server_tool_call"
                and unwrapped.get("name") != "code_interpreter"
            ):
                continue
            srv_step = _server_block_to_step(
                unwrapped,
                last_call_ids=last_call_ids,
                fallback_id=f"srv_{msg_idx}_{item_idx}",
                provider_kind=provider_kind,
            )
            if srv_step is not None:
                raw_srv_ex = unwrapped.get("extras")
                srv_thought_sig = (
                    _normalize_signature_b64(raw_srv_ex.get("thought_signature"))
                    if not is_foreign and isinstance(raw_srv_ex, Mapping)
                    else None
                )
                if seg_fcs or seg_contents:
                    _flush_segment()
                if (
                    srv_thought_sig
                    and srv_thought_sig not in _seg_thought_sigs()
                    and srv_thought_sig not in emitted_thought_sigs
                ):
                    if seg_server_steps or seg_thoughts or seg_carrier_sigs:
                        _flush_segment()
                    seg_thoughts.append(_build_thought_step(srv_thought_sig))
                seg_server_steps.append(srv_step)
            continue

        part_sig = _extract_trusted_signature(unwrapped, provider_kind=provider_kind)
        try:
            conv = _convert_block_to_interaction_content(
                unwrapped,
                use_vertexai=use_vertexai,
                model=effective_model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                provider_kind=provider_kind,
                allow_external_fetch=False,
                prefetched_urls=prefetched_urls,
            )
        except (ValueError, ChatGoogleGenerativeAIError) as exc:
            if provider_kind == "native":
                raise
            logger.warning("Dropping incompatible AI content block: %s", exc)
            continue

        if conv is None:
            if part_sig and btype in ("text", "text-plain", None):
                _flush_prior_tcs_for_new_sig(part_sig)
                if seg_fcs or seg_server_steps:
                    _flush_segment()
                if (
                    part_sig not in _seg_thought_sigs()
                    and part_sig not in emitted_thought_sigs
                ):
                    if (seg_thoughts or seg_carrier_sigs) and seg_contents:
                        _flush_segment()
                    seg_carrier_sigs.append(part_sig)
            continue

        if part_sig:
            _flush_prior_tcs_for_new_sig(part_sig)

        if seg_fcs or seg_server_steps:
            _flush_segment()

        if (
            part_sig
            and part_sig not in _seg_thought_sigs()
            and part_sig not in emitted_thought_sigs
        ):
            if (seg_thoughts or seg_carrier_sigs) and seg_contents:
                _flush_segment()
            seg_thoughts.append(_build_thought_step(part_sig))

        if conv.get("type") == "text" and not conv.get("text"):
            continue
        seg_contents.append(conv)

    # Emit any remaining msg.tool_calls not already placed inline
    for tc_idx in range(len(normalized_tcs)):
        if tc_idx not in consumed_tc_indices:
            _append_fc(tc_idx)

    _flush_segment()

    if not steps_out:
        return []

    # Gemini 3/4+ active-loop bypass signature per unsigned function_call segment
    if in_active_loop and _is_gemini_3_or_later(effective_model):
        idx_s = 0
        seg_start = 0
        seg_has_thought = False
        in_fc_run = False
        while idx_s < len(steps_out):
            st = steps_out[idx_s].get("type")
            if st == "thought":
                if in_fc_run:
                    seg_start = idx_s
                    in_fc_run = False
                seg_has_thought = True
                idx_s += 1
            elif st == "model_output":
                if in_fc_run:
                    seg_start = idx_s
                    seg_has_thought = False
                    in_fc_run = False
                idx_s += 1
            elif st in _RAW_SERVER_STEP_TYPES:
                seg_start = idx_s + 1
                seg_has_thought = False
                in_fc_run = False
                idx_s += 1
            elif st == "function_call":
                if not in_fc_run and not seg_has_thought:
                    steps_out.insert(
                        seg_start,
                        {"type": "thought", "signature": _SKIP_THOUGHT_SIGNATURE_B64},
                    )
                    seg_has_thought = True
                    in_fc_run = True
                    idx_s += 2
                else:
                    in_fc_run = True
                    idx_s += 1
            else:
                idx_s += 1

    return steps_out


def _messages_to_interaction_steps(
    messages: Sequence[BaseMessage],
    *,
    effective_model: str,
    use_vertexai: bool = False,
    convert_system_message_to_human: bool = False,
    default_resolution: Any = None,
    prior_messages: Sequence[BaseMessage] | None = None,
    prefetched_urls: Mapping[str, tuple[str, str | None]] | None = None,
    allow_empty_input: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate LangChain messages into `(system_instruction, steps)`."""
    if convert_system_message_to_human:
        warnings.warn(
            "The 'convert_system_message_to_human' parameter is deprecated and will be "
            "removed in a future version. Use system instructions instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    delta_is_system_only = bool(messages) and all(
        isinstance(m, SystemMessage) for m in messages
    )
    prior_had_human = prior_messages is not None and any(
        _is_real_user_prompt(m) for m in prior_messages
    )

    promoted_delta_sys: str | None = None
    if delta_is_system_only and prior_messages and not allow_empty_input:
        # Stateful delta contains only trailing SystemMessage(s): keep prior system
        # instruction intact and promote the delta SystemMessage(s) into user_input.
        system_instruction = (
            None
            if convert_system_message_to_human and prior_had_human
            else _extract_system_text(prior_messages)
        )
        promoted_delta_sys = _extract_system_text(messages)
    elif convert_system_message_to_human and prior_had_human:
        # Turn 1's SystemMessage is already embedded in Turn 1's stored HumanMessage.
        system_instruction = _extract_system_text(messages)
    else:
        all_for_sys = [*prior_messages, *messages] if prior_messages else messages
        system_instruction = _extract_system_text(all_for_sys)

    image_loader = ImageBytesLoader()
    known_call_names: dict[str, str] = {}
    pending_unided_calls: list[tuple[str, str]] = []

    # Seed known tool call names from prior_messages (when stateful slicing happened
    # right between AIMessage(tool_calls=...) and ToolMessage(...))!
    if prior_messages:
        for p_idx, p_msg in enumerate(prior_messages):
            if isinstance(p_msg, AIMessage) and p_msg.tool_calls:
                for tc_i, tc in enumerate(p_msg.tool_calls):
                    tc_id = tc.get("id")
                    tc_nm = tc.get("name") or ""
                    if tc_id and tc_nm:
                        known_call_names[str(tc_id)] = str(tc_nm)
                    elif not tc_id and tc_nm:
                        det = f"call_prior_{p_idx}_{tc_i}_{tc_nm}"
                        known_call_names[det] = str(tc_nm)
                        pending_unided_calls.append((str(tc_nm), det))

    active_loop_start = -1
    for idx in range(len(messages) - 1, -1, -1):
        if _is_real_user_prompt(messages[idx]):
            active_loop_start = idx
            break

    steps: list[dict[str, Any]] = []
    pending_tool_followup_media: list[dict[str, Any]] = []
    system_prepended = False

    def _flush_tool_followup_media() -> None:
        if pending_tool_followup_media:
            steps.append(
                {
                    "type": "user_input",
                    "content": list(pending_tool_followup_media),
                }
            )
            pending_tool_followup_media.clear()

    for idx, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            if not msg.content:
                if not allow_empty_input:
                    warnings.warn(
                        "HumanMessage with empty content was removed to prevent "
                        "API error",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            raw_h_items = (
                [msg.content] if isinstance(msg.content, str) else list(msg.content)
            )
            user_blocks: list[dict[str, Any]] = []
            has_non_tool_result = any(
                not (isinstance(it, Mapping) and it.get("type") == "tool_result")
                for it in raw_h_items
            )
            can_prepend_sys = (
                has_non_tool_result
                and convert_system_message_to_human
                and bool(system_instruction)
                and not system_prepended
            )

            for h_item in raw_h_items:
                # Support Anthropic `tool_result` blocks embedded in HumanMessage
                if isinstance(h_item, Mapping) and h_item.get("type") == "tool_result":
                    if user_blocks:
                        _flush_tool_followup_media()
                        steps.append({"type": "user_input", "content": user_blocks})
                        user_blocks = []
                    tr_id = str(h_item.get("tool_use_id") or f"call_{idx}")
                    tr_name = known_call_names.get(tr_id)
                    tr_step, tr_media = _convert_tool_result_payload(
                        h_item.get("content", ""),
                        call_id=tr_id,
                        name=tr_name,
                        is_error=bool(h_item.get("is_error")),
                        use_vertexai=use_vertexai,
                        model=effective_model,
                        default_resolution=default_resolution,
                        image_loader=image_loader,
                        prefetched_urls=prefetched_urls,
                    )
                    steps.append(tr_step)
                    if tr_media:
                        pending_tool_followup_media.extend(tr_media)
                    continue

                conv_h = _convert_block_to_interaction_content(
                    h_item,
                    use_vertexai=use_vertexai,
                    model=effective_model,
                    default_resolution=default_resolution,
                    image_loader=image_loader,
                    allow_external_fetch=True,
                    prefetched_urls=prefetched_urls,
                )
                if conv_h is not None:
                    _flush_tool_followup_media()
                    if can_prepend_sys and system_instruction and not system_prepended:
                        user_blocks.append({"type": "text", "text": system_instruction})
                        system_instruction = None
                        system_prepended = True
                    user_blocks.append(conv_h)

            if user_blocks:
                _flush_tool_followup_media()
                steps.append({"type": "user_input", "content": user_blocks})
            elif (
                not allow_empty_input
                and has_non_tool_result
                and all(
                    (isinstance(it, str) and not it)
                    or (isinstance(it, Mapping) and _is_empty_human_text_block(it))
                    for it in raw_h_items
                )
            ):
                warnings.warn(
                    "HumanMessage with empty content was removed to prevent API error",
                    UserWarning,
                    stacklevel=2,
                )
            continue

        if isinstance(msg, AIMessage):
            _flush_tool_followup_media()
            in_active = idx > active_loop_start if active_loop_start != -1 else True
            steps.extend(
                _convert_ai_message_to_steps(
                    msg,
                    msg_idx=idx,
                    in_active_loop=in_active,
                    effective_model=effective_model,
                    use_vertexai=use_vertexai,
                    default_resolution=default_resolution,
                    image_loader=image_loader,
                    known_call_names=known_call_names,
                    pending_unided_calls=pending_unided_calls,
                    prefetched_urls=prefetched_urls,
                )
            )
            continue

        if isinstance(msg, (ToolMessage, FunctionMessage)):
            explicit_name = msg.name or msg.additional_kwargs.get("name") or None
            raw_call_id = msg.tool_call_id if isinstance(msg, ToolMessage) else None
            if isinstance(raw_call_id, str) and raw_call_id:
                call_id = raw_call_id
            else:
                matched_det: str | None = None
                if explicit_name:
                    for p_i, (p_nm, p_id) in enumerate(pending_unided_calls):
                        if p_nm == explicit_name:
                            matched_det = p_id
                            pending_unided_calls.pop(p_i)
                            break
                if matched_det is None and pending_unided_calls:
                    _, matched_det = pending_unided_calls.pop(0)
                call_id = matched_det or f"call_{idx}_{explicit_name or 'tool'}"

            tool_name = explicit_name or known_call_names.get(call_id)
            is_err = getattr(msg, "status", None) == "error"
            fr_step, fr_media = _convert_tool_result_payload(
                msg.content,
                call_id=call_id,
                name=str(tool_name) if tool_name else None,
                is_error=is_err,
                use_vertexai=use_vertexai,
                model=effective_model,
                default_resolution=default_resolution,
                image_loader=image_loader,
                prefetched_urls=prefetched_urls,
            )
            steps.append(fr_step)
            if fr_media:
                pending_tool_followup_media.extend(fr_media)
            continue

        msg_err = f"Unexpected message with type {type(msg)} at position {idx}."
        raise GoogleInvalidRequestError(msg_err)

    _flush_tool_followup_media()

    if not steps and promoted_delta_sys:
        steps.append(
            {
                "type": "user_input",
                "content": [{"type": "text", "text": promoted_delta_sys}],
            }
        )
    elif not steps and not allow_empty_input:
        if prior_messages:
            delta_only_sys = _extract_system_text(messages)
            if delta_only_sys:
                steps.append(
                    {
                        "type": "user_input",
                        "content": [{"type": "text", "text": delta_only_sys}],
                    }
                )
                system_instruction = (
                    None
                    if convert_system_message_to_human and prior_had_human
                    else _extract_system_text(prior_messages)
                )
        elif system_instruction:
            steps.append(
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": system_instruction}],
                }
            )
            system_instruction = None

    if not steps:
        if allow_empty_input:
            return system_instruction, []
        msg_empty = "Cannot create an interaction with empty input."
        raise GoogleInvalidRequestError(msg_empty)

    if _uses_fixed_sampling_and_disallows_prefill(effective_model) and steps[-1].get(
        "type"
    ) not in ("user_input", "function_result"):
        msg_prefill = (
            f"Model '{effective_model}' does not support model prefilling. The final "
            "request turn must be a user message or a function response."
        )
        raise GoogleInvalidRequestError(msg_prefill)

    return system_instruction, steps


def _has_replayed_model_output_and_function_call(
    steps: Sequence[Mapping[str, Any]],
) -> bool:
    """Detect assistant turns containing both `model_output` and `function_call`."""
    saw_model_output = False
    for step in steps:
        stype = step.get("type")
        if stype == "model_output":
            saw_model_output = True
        elif stype == "function_call" and saw_model_output:
            return True
        elif stype in ("user_input", "function_result"):
            saw_model_output = False
    return False


def _is_custom_gateway_url(base_url: str | None) -> bool:
    """Return True when `base_url` points to a non-`googleapis.com` custom gateway."""
    if not base_url or not isinstance(base_url, str):
        return False
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return False
    return host != "googleapis.com" and not host.endswith(".googleapis.com")


def _resolve_interactions_api_version(
    model_instance: Any,
    per_request_api_version: str | None = None,
    *,
    per_request_base_url: str | None = None,
) -> str | None:
    """Guard `api_version='v1'` on native Google hosts while preserving gateways."""
    use_vertexai = bool(getattr(model_instance, "_use_vertexai", False))
    api_client = getattr(getattr(model_instance, "client", None), "_api_client", None)
    project = getattr(model_instance, "project", None) or getattr(
        api_client, "project", None
    )
    location = (
        getattr(model_instance, "location", None)
        or getattr(api_client, "location", None)
        or "global"
    )

    inst_base = getattr(model_instance, "base_url", None)
    if isinstance(inst_base, Mapping):
        inst_base = inst_base.get("api_endpoint")
    client_base = getattr(getattr(api_client, "_http_options", None), "base_url", None)
    eff_base = per_request_base_url or (
        str(inst_base)
        if isinstance(inst_base, str) and inst_base
        else str(client_base)
        if isinstance(client_base, str) and client_base
        else None
    )
    is_custom_gateway = _is_custom_gateway_url(eff_base)

    candidate = per_request_api_version or getattr(model_instance, "api_version", None)
    if candidate == "v1" and not is_custom_gateway:
        if use_vertexai:
            if project:
                return f"v1beta1/projects/{project}/locations/{location}"
            return "v1beta1"
        return "v1beta"
    if per_request_api_version:
        if use_vertexai and project and "/projects/" not in per_request_api_version:
            return f"{per_request_api_version}/projects/{project}/locations/{location}"
        return per_request_api_version
    if is_custom_gateway:
        return None
    # Also guard Vertex clients whose location was unset on mock/custom clients
    if (
        use_vertexai
        and project
        and api_client is not None
        and not getattr(api_client, "location", None)
    ):
        ver = (
            getattr(getattr(api_client, "_http_options", None), "api_version", None)
            or "v1beta1"
        )
        return f"{ver}/projects/{project}/locations/{location}"
    return None


def _prepare_interactions_request(
    model_instance: Any,
    messages: Sequence[BaseMessage],
    *,
    stop: list[str] | None = None,
    tools: Sequence[Any] | None = None,
    functions: Sequence[Any] | None = None,
    safety_settings: SafetySettingDict | Sequence[Any] | None = None,
    tool_config: Mapping[str, Any] | ToolConfig | None = None,
    generation_config: Mapping[str, Any] | None = None,
    cached_content: str | None = None,
    tool_choice: _ToolChoiceType | bool | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> _InteractionsPreparedRequest:
    """Build the full keyword argument payload for `client.interactions.create(...)`."""
    kw = dict(kwargs)
    mk = dict(getattr(model_instance, "model_kwargs", None) or {})
    gen_dict = dict(generation_config) if generation_config else {}
    use_vertexai = bool(getattr(model_instance, "_use_vertexai", False))

    kw.pop("strict", None)
    kw.pop("use_interactions_api", None)
    output_version = kw.pop("output_version", None) or getattr(
        model_instance, "output_version", None
    )

    # Handle direct `code_execution=True` kwarg if passed outside `invoke()`
    code_exec_flag = kw.pop("code_execution", None)
    effective_tools: list[Any] | None = list(tools) if tools is not None else None
    if effective_tools is None and "tools" in mk:
        effective_tools = list(mk["tools"])
    if code_exec_flag:
        if not model_instance._supports_code_execution:
            msg = f"Code execution is not supported on model: {model_instance.model}"
            raise GoogleInvalidRequestError(msg)
        if effective_tools:
            msg = "Tools are already defined. code_execution tool can't be defined."
            raise ValueError(msg)
        from google.genai.types import ToolCodeExecution

        effective_tools = [GoogleTool(code_execution=ToolCodeExecution())]

    # Resolve model vs agent
    raw_model_override = kw.pop("model", None) or mk.get("model")
    raw_model = str(raw_model_override or model_instance.model)
    effective_model = _strip_model_prefix(raw_model)

    raw_agent = (
        kw.pop("agent", None)
        or getattr(model_instance, "agent", None)
        or mk.get("agent")
    )
    effective_agent = _strip_model_prefix(str(raw_agent)) if raw_agent else None
    raw_agent_config = (
        kw.pop("agent_config", None)
        or getattr(model_instance, "agent_config", None)
        or mk.get("agent_config")
    )
    if effective_agent is None and _normalize_gemini_model_name(
        effective_model
    ).startswith(_INTERACTIONS_AUTO_MODEL_PREFIXES):
        effective_agent = effective_model
    is_agent_run = effective_agent is not None or raw_agent_config is not None

    # Stateful previous_interaction_id slicing (respecting explicit store=False)
    use_prev_kw = kw.pop("use_previous_interaction_id", None)
    mk_prev = mk.get("use_previous_interaction_id")
    use_prev_flag: bool = (
        use_prev_kw
        if isinstance(use_prev_kw, bool)
        else mk_prev
        if isinstance(mk_prev, bool)
        else bool(getattr(model_instance, "use_previous_interaction_id", False))
    )
    explicit_prev_id = kw.pop("previous_interaction_id", None) or mk.get(
        "previous_interaction_id"
    )
    call_store = kw.pop("store", None)
    explicit_store = _resolve_explicit_store(
        call_store=call_store,
        model_kwargs_store=mk.get("store"),
        instance_store=model_instance.store,
    )
    raw_cont_tok = kw.pop("continuation_token", None)
    if raw_cont_tok is None:
        raw_cont_tok = mk.get("continuation_token")
    clean_cont_tok = _clean_continuation_token(raw_cont_tok)

    prev_id, prior_msgs, active_messages = _resolve_stateful_slicing(
        messages,
        use_prev_flag=use_prev_flag,
        explicit_prev_id=explicit_prev_id,
        explicit_store=explicit_store,
        continuation_token=clean_cont_tok,
    )

    raw_bg = kw.pop("background", None)
    if raw_bg is None:
        raw_bg = (
            model_instance.background
            if model_instance.background is not None
            else mk.get("background")
        )
    if (
        raw_bg is None
        and effective_agent is not None
        and _normalize_gemini_model_name(effective_agent).startswith("deep-research-")
    ):
        raw_bg = True
    effective_bg = bool(raw_bg) if raw_bg is not None else False

    effective_store = _resolve_store(
        call_store=call_store,
        model_kwargs_store=mk.get("store"),
        instance_store=model_instance.store,
        use_previous_interaction_id=use_prev_flag,
        previous_interaction_id=prev_id,
        background=effective_bg,
        agent=effective_agent,
    )

    automatic_continuation = kw.pop(
        "automatic_continuation", mk.get("automatic_continuation")
    )
    if effective_bg and automatic_continuation is False:
        msg_ac = "automatic_continuation=False cannot be used with background=True."
        raise GoogleInvalidRequestError(msg_ac)

    raw_pi = kw.pop("polling_interval", None)
    if raw_pi is None:
        raw_pi = (
            model_instance.polling_interval
            if model_instance.polling_interval is not None
            else mk.get("polling_interval")
        )
    polling_interval = (
        float(raw_pi) if raw_pi is not None else _DEFAULT_POLLING_INTERVAL_S
    )
    if polling_interval <= 0:
        msg_pi = "polling_interval must be positive"
        raise ValueError(msg_pi)

    raw_poll = kw.pop("poll", mk.get("poll", True))
    should_poll = True if raw_poll is None else bool(raw_poll)

    # Timeout (in seconds, default 600.0), retries, headers, base_url, api_version
    raw_http_opts = kw.pop("http_options", None)
    per_req_http: HttpOptions | None = None
    if isinstance(raw_http_opts, Mapping):
        per_req_http = HttpOptions(**raw_http_opts)
    elif isinstance(raw_http_opts, HttpOptions):
        per_req_http = raw_http_opts

    per_req_base_url = (
        str(per_req_http.base_url).strip()
        if per_req_http is not None
        and per_req_http.base_url is not None
        and str(per_req_http.base_url).strip()
        else None
    )
    per_req_client_args = (
        dict(per_req_http.client_args)
        if per_req_http is not None and per_req_http.client_args
        else None
    )
    per_req_async_client_args = (
        dict(per_req_http.async_client_args)
        if per_req_http is not None and per_req_http.async_client_args
        else (per_req_client_args if per_req_client_args is not None else None)
    )
    per_req_httpx_client = (
        per_req_http.httpx_client if per_req_http is not None else None
    )
    per_req_httpx_async_client = (
        per_req_http.httpx_async_client if per_req_http is not None else None
    )

    call_timeout = kw.pop("timeout", None)
    resolved_timeout: Any = _DEFAULT_INTERACTIONS_TIMEOUT_S
    timeout_sec_float: float | None = _DEFAULT_INTERACTIONS_TIMEOUT_S
    if isinstance(call_timeout, _HTTPX_TIMEOUT_TYPES):
        resolved_timeout = call_timeout
        timeout_sec_float = call_timeout.read
    elif per_req_http is not None and per_req_http.timeout is not None:
        timeout_sec_float = float(per_req_http.timeout) / 1000.0
        resolved_timeout = timeout_sec_float
    elif call_timeout is not None:
        timeout_sec_float = float(call_timeout)
        resolved_timeout = timeout_sec_float
    elif model_instance.timeout is not None:
        timeout_sec_float = float(model_instance.timeout)
        resolved_timeout = timeout_sec_float

    is_async_call = bool(kw.pop("_is_async", False))
    prefetched_urls = kw.pop("_prefetched_urls", None)
    if not is_async_call and prefetched_urls is None:
        sync_urls = _collect_external_fetch_urls(active_messages)
        if sync_urls:
            sync_loader = ImageBytesLoader()
            borrowed_sync, fetch_sync_kw = _build_sync_prefetch_client_kwargs(
                model_instance, per_req_http, call_timeout=call_timeout
            )
            explicit_ft = _resolve_media_fetch_timeout(
                model_instance, per_req_http, call_timeout
            )
            if borrowed_sync is not None:
                enforce_redir = not bool(
                    getattr(borrowed_sync, "follow_redirects", False)
                )
                prefetched_urls = {
                    u: _fetch_with_sync_client(
                        u,
                        borrowed_sync,
                        sync_loader,
                        request_timeout=explicit_ft,
                        enforce_follow_redirects=enforce_redir,
                    )
                    for u in sync_urls
                }
            else:
                with _create_httpx_client(fetch_sync_kw, is_async=False) as sync_http:
                    prefetched_urls = {
                        u: _fetch_with_sync_client(u, sync_http, sync_loader)
                        for u in sync_urls
                    }

    explicit_sys = kw.pop("system_instruction", None)
    system_instruction, steps = _messages_to_interaction_steps(
        active_messages,
        effective_model=effective_model,
        use_vertexai=use_vertexai,
        convert_system_message_to_human=bool(
            getattr(model_instance, "convert_system_message_to_human", False)
        ),
        default_resolution=kw.pop(
            "media_resolution", getattr(model_instance, "media_resolution", None)
        ),
        prior_messages=prior_msgs,
        prefetched_urls=prefetched_urls
        if isinstance(prefetched_urls, Mapping)
        else None,
        allow_empty_input=clean_cont_tok is not None,
    )
    if isinstance(explicit_sys, str) and explicit_sys:
        system_instruction = (
            f"{explicit_sys}\n\n{system_instruction}"
            if system_instruction
            else explicit_sys
        )

    # Vertex AI 400/timeout guard when caller forces store=True without
    # previous_interaction_id on replayed [model_output, function_call] turns.
    if (
        use_vertexai
        and effective_store
        and not prev_id
        and not effective_bg
        and _has_replayed_model_output_and_function_call(steps)
    ):
        warnings.warn(
            "Vertex AI Interactions API rejects store=True without "
            "previous_interaction_id when replaying history containing "
            "[model_output, function_call] in one turn; downgrading store=False.",
            UserWarning,
            stacklevel=2,
        )
        effective_store = False

    # Warn & omit cached_content
    eff_cache = (
        cached_content
        or kw.pop("cached_content", None)
        or getattr(model_instance, "cached_content", None)
    )
    if eff_cache is not None:
        warnings.warn(
            "Gemini Interactions API deprecates/omits 'cached_content'; "
            "use stateful previous_interaction_id or implicit caching instead.",
            UserWarning,
            stacklevel=2,
        )

    formatted_tools = _format_interactions_tools(
        effective_tools, functions, tool_config=tool_config
    )
    tool_choice_param = _format_interactions_tool_choice(
        tool_choice, tool_config, formatted_tools
    )

    eff_safety = (
        safety_settings
        if safety_settings is not None
        else kw.pop("safety_settings", model_instance.safety_settings)
    )
    formatted_safety = _format_interactions_safety_settings(eff_safety)

    # Response format & multimodal outputs
    rf_kwarg = kw.pop("response_format", None)
    resp_mime = kw.pop(
        "response_mime_type",
        gen_dict.pop("response_mime_type", model_instance.response_mime_type),
    )
    resp_schema = kw.pop(
        "response_schema",
        gen_dict.pop("response_schema", model_instance.response_schema),
    )
    resp_json_schema = kw.pop(
        "response_json_schema", gen_dict.pop("response_json_schema", None)
    )
    resp_modalities = kw.pop(
        "response_modalities",
        gen_dict.pop("response_modalities", model_instance.response_modalities),
    )
    img_cfg = kw.pop(
        "image_config", gen_dict.pop("image_config", model_instance.image_config)
    )
    structured_schema_hint = kw.pop("_structured_output_schema", None)
    kw.pop("ls_structured_output_format", None)
    response_format_param, pydantic_schema = _format_interactions_response_format(
        effective_model=effective_model,
        response_format_kwarg=rf_kwarg,
        response_mime_type=resp_mime,
        response_schema=resp_schema,
        response_json_schema=resp_json_schema,
        response_modalities=resp_modalities,
        image_config=img_cfg,
        pydantic_schema_hint=(
            structured_schema_hint if isinstance(structured_schema_hint, type) else None
        ),
    )

    max_retries = kw.pop("max_retries", None)
    if max_retries is None:
        max_retries = model_instance.max_retries

    merged_headers: dict[str, str] = {}
    if per_req_http is not None and per_req_http.headers:
        merged_headers.update(per_req_http.headers)
    call_extra_headers = kw.pop("extra_headers", None)
    if isinstance(call_extra_headers, Mapping):
        merged_headers.update(call_extra_headers)

    per_req_api_ver = kw.pop("api_version", None) or (
        per_req_http.api_version if per_req_http is not None else None
    )
    resolved_api_version = _resolve_interactions_api_version(
        model_instance, per_req_api_ver, per_request_base_url=per_req_base_url
    )

    caller_extra_body = kw.pop("extra_body", None)
    extra_body: dict[str, Any] = (
        copy.deepcopy(dict(caller_extra_body))
        if isinstance(caller_extra_body, Mapping)
        else {}
    )
    extra_query = kw.pop("extra_query", None)

    labels = kw.pop("labels", None) or model_instance.labels
    raw_service_tier = (
        kw.pop("service_tier", None)
        or model_instance.service_tier
        or mk.get("service_tier")
    )
    service_tier = (
        str(_unwrap_enum_value(raw_service_tier)).lower()
        if raw_service_tier is not None
        else None
    )
    environment = (
        kw.pop("environment", None)
        or model_instance.environment
        or mk.get("environment")
    )
    webhook_config = kw.pop("webhook_config", None) or mk.get("webhook_config")

    # Generation config (model interactions only — omitted on CreateAgentInteraction)
    gen_cfg_param: dict[str, Any] | None = None
    if not is_agent_run:
        # Resolve effective temperature respecting Gemini 3+ default None
        if "temperature" in kw:
            eff_temp = kw.pop("temperature")
        elif "temperature" in gen_dict:
            eff_temp = gen_dict.pop("temperature")
        elif "temperature" in model_instance.model_fields_set:
            eff_temp = model_instance.temperature
        elif _is_gemini_3_or_later(effective_model):
            eff_temp = None
        else:
            eff_temp = model_instance.temperature

        eff_top_p = kw.pop("top_p", gen_dict.pop("top_p", model_instance.top_p))
        eff_top_k = kw.pop("top_k", gen_dict.pop("top_k", model_instance.top_k))
        eff_freq = kw.pop(
            "frequency_penalty",
            gen_dict.pop("frequency_penalty", model_instance.frequency_penalty),
        )
        eff_pres = kw.pop(
            "presence_penalty",
            gen_dict.pop("presence_penalty", model_instance.presence_penalty),
        )
        eff_max_tok = kw.pop(
            "max_output_tokens",
            kw.pop(
                "max_tokens",
                gen_dict.pop("max_output_tokens", model_instance.max_output_tokens),
            ),
        )
        eff_seed = kw.pop("seed", gen_dict.pop("seed", model_instance.seed))
        eff_n = kw.pop(
            "n",
            kw.pop(
                "candidate_count", gen_dict.pop("candidate_count", model_instance.n)
            ),
        )
        eff_stop = (
            stop
            if stop is not None
            else kw.pop(
                "stop_sequences", gen_dict.pop("stop_sequences", model_instance.stop)
            )
        )

        raw_tc = kw.pop(
            "thinking_config",
            gen_dict.pop("thinking_config", model_instance.thinking_config),
        )
        t_lvl_in = kw.pop(
            "thinking_level",
            gen_dict.pop(
                "thinking_level",
                kw.pop("reasoning_effort", model_instance.reasoning_effort),
            ),
        )
        t_bud_in = kw.pop(
            "thinking_budget",
            gen_dict.pop("thinking_budget", model_instance.thinking_budget),
        )
        inc_th_in = kw.pop(
            "include_thoughts",
            gen_dict.pop("include_thoughts", model_instance.include_thoughts),
        )
        t_sum_in = kw.pop(
            "thinking_summaries", gen_dict.pop("thinking_summaries", None)
        )

        eff_profile = model_instance.profile if raw_model_override is None else None
        t_level, t_summaries = _resolve_interactions_thinking(
            effective_model=effective_model,
            profile=eff_profile,
            raw_thinking_config=raw_tc,
            thinking_level=t_lvl_in,
            thinking_budget=t_bud_in,
            include_thoughts=inc_th_in,
            thinking_summaries=t_sum_in,
        )

        norm_m = _normalize_gemini_model_name(effective_model)
        is_tts = norm_m.endswith("-tts") or "-tts-" in norm_m
        speech_cfg = _format_interactions_speech_config(
            kw.pop("speech_config", gen_dict.pop("speech_config", None)),
            voice_name=kw.pop("voice_name", gen_dict.pop("voice_name", None)),
            is_tts_model=is_tts,
        )
        if tool_choice_param is None and "tool_choice" in gen_dict:
            tool_choice_param = gen_dict.pop("tool_choice")

        gen_cfg_param, extra_gen_cfg = _build_interactions_generation_config(
            effective_model=effective_model,
            temperature=eff_temp,
            top_p=eff_top_p,
            top_k=eff_top_k,
            frequency_penalty=eff_freq,
            presence_penalty=eff_pres,
            max_output_tokens=eff_max_tok,
            seed=eff_seed,
            candidate_count=eff_n,
            stop_sequences=eff_stop,
            thinking_level=t_level,
            thinking_summaries=t_summaries,
            tool_choice_param=tool_choice_param,
            speech_config=speech_cfg,
            transcription_config=kw.pop(
                "transcription_config", gen_dict.pop("transcription_config", None)
            ),
            video_config=kw.pop("video_config", gen_dict.pop("video_config", None)),
            logprobs=kw.pop("logprobs", gen_dict.pop("logprobs", None)),
            response_logprobs=kw.pop(
                "response_logprobs", gen_dict.pop("response_logprobs", None)
            ),
        )
        if extra_gen_cfg:
            # NOTE: Speakeasy serialize_request_body merges extra_body via shallow
            # dict.update(), so extra_body["generation_config"] must preserve every
            # field from gen_cfg_param alongside temperature/top_p!
            existing_eb_gc = (
                dict(extra_body["generation_config"])
                if isinstance(extra_body.get("generation_config"), Mapping)
                else {}
            )
            extra_body["generation_config"] = {
                **(gen_cfg_param or {}),
                **extra_gen_cfg,
                **existing_eb_gc,
            }
    else:
        # Consume model-only generation kwargs silently when running an agent
        for drop_k in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "max_output_tokens",
            "max_tokens",
            "seed",
            "n",
            "candidate_count",
            "stop_sequences",
            "thinking_config",
            "thinking_level",
            "reasoning_effort",
            "thinking_budget",
            "include_thoughts",
            "thinking_summaries",
            "speech_config",
            "voice_name",
            "transcription_config",
            "video_config",
            "logprobs",
            "response_logprobs",
        ):
            kw.pop(drop_k, None)

    create_kwargs: dict[str, Any] = {
        "store": effective_store,
        "timeout": resolved_timeout,
    }
    if steps:
        create_kwargs["input"] = steps
    if stream:
        create_kwargs["stream"] = True
    if is_agent_run:
        target_agent = effective_agent or effective_model
        create_kwargs["agent"] = target_agent
        agent_cfg = _format_interactions_agent_config(target_agent, raw_agent_config)
        if agent_cfg is not None:
            create_kwargs["agent_config"] = agent_cfg
        target_name = target_agent
    else:
        create_kwargs["model"] = effective_model
        if gen_cfg_param is not None:
            create_kwargs["generation_config"] = gen_cfg_param
        target_name = effective_model

    if system_instruction is not None:
        create_kwargs["system_instruction"] = system_instruction
    if formatted_tools is not None:
        create_kwargs["tools"] = formatted_tools
    if response_format_param is not None:
        create_kwargs["response_format"] = response_format_param
    if formatted_safety is not None:
        create_kwargs["safety_settings"] = formatted_safety
    if prev_id is not None:
        create_kwargs["previous_interaction_id"] = prev_id
    if raw_bg is not None:
        create_kwargs["background"] = effective_bg
    if labels is not None:
        create_kwargs["labels"] = dict(labels)
    if service_tier is not None:
        create_kwargs["service_tier"] = service_tier
    if environment is not None:
        create_kwargs["environment"] = copy.deepcopy(environment)
    if webhook_config is not None:
        create_kwargs["webhook_config"] = copy.deepcopy(webhook_config)
    if clean_cont_tok is not None:
        if _HAS_NATIVE_CONTINUATION_TOKEN:
            create_kwargs["continuation_token"] = clean_cont_tok
        else:
            extra_body.setdefault("continuation_token", clean_cont_tok)
    if automatic_continuation is not None:
        if _resolve_create_supports_automatic_continuation(
            model_instance, is_async=is_async_call
        ):
            create_kwargs["automatic_continuation"] = bool(automatic_continuation)
        elif bool(automatic_continuation):
            warnings.warn(
                "Installed 'google-genai' version does not support client-side "
                "'automatic_continuation'; upgrade to 'google-genai>=2.34.0'.",
                UserWarning,
                stacklevel=2,
            )

    if kw:
        logger.warning(
            "Unrecognized keyword argument(s) ignored for Gemini Interactions API: %s",
            sorted(kw.keys()),
        )

    if resolved_api_version is not None:
        create_kwargs["api_version"] = resolved_api_version
    if merged_headers:
        create_kwargs["extra_headers"] = merged_headers
    if isinstance(extra_query, Mapping) and extra_query:
        create_kwargs["extra_query"] = dict(extra_query)
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    return _InteractionsPreparedRequest(
        create_kwargs=create_kwargs,
        target_name=target_name,
        max_retries=max_retries,
        requested_store=effective_store,
        background=effective_bg,
        polling_interval=polling_interval,
        timeout_seconds=timeout_sec_float,
        api_version=resolved_api_version,
        extra_headers=merged_headers or None,
        output_version=output_version,
        pydantic_schema=pydantic_schema,
        poll=should_poll,
        previous_interaction_id=prev_id,
        base_url_override=per_req_base_url,
        client_args_override=per_req_client_args,
        async_client_args_override=per_req_async_client_args,
        httpx_client_override=per_req_httpx_client,
        httpx_async_client_override=per_req_httpx_async_client,
    )


def _resolve_create_supports_automatic_continuation(
    model_instance: Any, *, is_async: bool = False
) -> bool:
    """Return True when the active SDK/mock `interactions.create` accepts the kwarg."""
    client = getattr(
        model_instance, "async_client" if is_async else "client", None
    ) or getattr(model_instance, "client", None)
    if client is None:
        return False
    injected = getattr(client, "_interactions", None)
    cls_ia = getattr(type(client), "interactions", None)
    has_patched = "interactions" in getattr(client, "__dict__", {}) or (
        cls_ia is not None and not isinstance(cls_ia, property)
    )
    if (
        has_patched
        or not type(client).__module__.startswith("google.genai.")
        or (
            injected is not None
            and not type(injected).__module__.startswith("google.genai.")
        )
    ):
        target_ia = (
            getattr(client, "interactions", None)
            if has_patched or injected is None
            else injected
        )
        create_fn = getattr(target_ia, "create", None)
        if not callable(create_fn):
            return False
        try:
            params = inspect.signature(create_fn).parameters
        except (TypeError, ValueError):
            return True
        return "automatic_continuation" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    try:
        if is_async:
            from google.genai._gaos.google_genai import AsyncGeminiNextGenInteractions

            return (
                "automatic_continuation"
                in inspect.signature(AsyncGeminiNextGenInteractions.create).parameters
            )
        from google.genai._gaos.google_genai import GeminiNextGenInteractions

        return (
            "automatic_continuation"
            in inspect.signature(GeminiNextGenInteractions.create).parameters
        )
    except Exception:
        return False


def _coerce_http_options(raw_http_opts: Any) -> HttpOptions | None:
    """Normalize a mapping or `HttpOptions` instance to `HttpOptions | None`."""
    if isinstance(raw_http_opts, Mapping):
        return HttpOptions(**raw_http_opts)
    if isinstance(raw_http_opts, HttpOptions):
        return raw_http_opts
    return None


def _resolve_media_fetch_timeout(
    model_instance: Any,
    per_req_http: HttpOptions | None,
    call_timeout: Any,
) -> Any:
    """Resolve an explicit finite media-fetch timeout from request/model settings."""
    if isinstance(call_timeout, _HTTPX_TIMEOUT_TYPES):
        if call_timeout.read is None:
            return type(call_timeout)(
                60.0,
                connect=call_timeout.connect
                if call_timeout.connect is not None
                else 60.0,
                write=call_timeout.write if call_timeout.write is not None else 60.0,
                pool=call_timeout.pool if call_timeout.pool is not None else 60.0,
            )
        return call_timeout
    raw_sec: float | None = None
    if per_req_http is not None and per_req_http.timeout is not None:
        raw_sec = float(per_req_http.timeout) / 1000.0
    elif call_timeout is not None and not isinstance(call_timeout, bool):
        raw_sec = float(call_timeout)
    elif getattr(model_instance, "timeout", None) is not None:
        raw_sec = float(model_instance.timeout)
    if raw_sec is None:
        return None
    return raw_sec if raw_sec > 0 else 60.0


def _build_sync_prefetch_client_kwargs(
    model_instance: Any,
    raw_http_opts: Any,
    *,
    call_timeout: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve borrowed `httpx_client` or merged sync client kwargs."""
    per_req_http = _coerce_http_options(raw_http_opts)
    if (
        per_req_http is not None
        and per_req_http.httpx_client is not None
        and isinstance(per_req_http.httpx_client, _HTTPX_SYNC_CLIENT_TYPES)
    ):
        return per_req_http.httpx_client, {}

    override_args: dict[str, Any] = {}
    if per_req_http is not None and per_req_http.client_args:
        for k, v in per_req_http.client_args.items():
            if k == "transport" and v is not None:
                if not isinstance(v, _HTTPX_SYNC_TRANSPORT_TYPES):
                    continue
            override_args[k] = v

    merged = _merge_ephemeral_httpx_kwargs(
        getattr(model_instance, "client_args", None),
        override_args,
        is_async=False,
    )
    merged.pop("base_url", None)
    explicit_to = _resolve_media_fetch_timeout(
        model_instance, per_req_http, call_timeout
    )
    if explicit_to is not None:
        merged["timeout"] = explicit_to
    elif merged.get("timeout") is None:
        merged["timeout"] = 60.0
    merged.setdefault("follow_redirects", True)
    return None, merged


def _build_async_prefetch_client_kwargs(
    model_instance: Any,
    raw_http_opts: Any,
    *,
    call_timeout: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve borrowed `httpx_async_client` or merged async client kwargs."""
    per_req_http = _coerce_http_options(raw_http_opts)
    if (
        per_req_http is not None
        and per_req_http.httpx_async_client is not None
        and isinstance(per_req_http.httpx_async_client, _HTTPX_ASYNC_CLIENT_TYPES)
    ):
        return per_req_http.httpx_async_client, {}

    override_args: dict[str, Any] = {}
    if per_req_http is not None and per_req_http.client_args:
        for k, v in per_req_http.client_args.items():
            if k == "transport" and v is not None:
                if not isinstance(v, _HTTPX_ASYNC_TRANSPORT_TYPES):
                    continue
            override_args[k] = v
    if per_req_http is not None and per_req_http.async_client_args:
        override_args.update(dict(per_req_http.async_client_args))

    merged = _merge_ephemeral_httpx_kwargs(
        getattr(model_instance, "client_args", None),
        override_args,
        is_async=True,
    )
    merged.pop("base_url", None)
    explicit_to = _resolve_media_fetch_timeout(
        model_instance, per_req_http, call_timeout
    )
    if explicit_to is not None:
        merged["timeout"] = explicit_to
    elif merged.get("timeout") is None:
        merged["timeout"] = 60.0
    merged.setdefault("follow_redirects", True)
    return None, merged


async def _aprepare_interactions_request(
    model_instance: Any,
    messages: Sequence[BaseMessage],
    *,
    stop: list[str] | None = None,
    tools: Sequence[Any] | None = None,
    functions: Sequence[Any] | None = None,
    safety_settings: SafetySettingDict | Sequence[Any] | None = None,
    tool_config: Mapping[str, Any] | ToolConfig | None = None,
    generation_config: Mapping[str, Any] | None = None,
    cached_content: str | None = None,
    tool_choice: _ToolChoiceType | bool | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> _InteractionsPreparedRequest:
    """Prepare an Interactions request asynchronously without blocking asyncio."""
    mk = dict(getattr(model_instance, "model_kwargs", None) or {})
    use_prev_kw = kwargs.get("use_previous_interaction_id")
    mk_prev = mk.get("use_previous_interaction_id")
    use_prev_flag: bool = (
        use_prev_kw
        if isinstance(use_prev_kw, bool)
        else mk_prev
        if isinstance(mk_prev, bool)
        else bool(getattr(model_instance, "use_previous_interaction_id", False))
    )
    explicit_prev_id = kwargs.get("previous_interaction_id") or mk.get(
        "previous_interaction_id"
    )
    raw_cont_pre = kwargs.get("continuation_token")
    if raw_cont_pre is None:
        raw_cont_pre = mk.get("continuation_token")
    clean_cont_pre = _clean_continuation_token(raw_cont_pre)
    explicit_store = _resolve_explicit_store(
        call_store=kwargs.get("store"),
        model_kwargs_store=mk.get("store"),
        instance_store=model_instance.store,
    )
    prev_id_pre, _, active_messages = _resolve_stateful_slicing(
        messages,
        use_prev_flag=use_prev_flag,
        explicit_prev_id=explicit_prev_id,
        explicit_store=explicit_store,
        continuation_token=clean_cont_pre,
    )

    raw_bg_pre = kwargs.get("background")
    if raw_bg_pre is None:
        raw_bg_pre = (
            model_instance.background
            if model_instance.background is not None
            else mk.get("background")
        )
    if explicit_store is False and (
        bool(prev_id_pre)
        or (explicit_prev_id and str(explicit_prev_id).strip())
        or bool(raw_bg_pre)
    ):
        msg_inv = (
            "store=False cannot be combined with previous_interaction_id "
            "or background=True in the Gemini Interactions API."
        )
        raise GoogleInvalidRequestError(msg_inv)

    urls_to_fetch = _collect_external_fetch_urls(active_messages)
    prefetched: dict[str, tuple[str, str | None]] = {}
    if urls_to_fetch:
        image_loader = ImageBytesLoader()
        raw_http_pre = kwargs.get("http_options")
        call_to_pre = kwargs.get("timeout")
        borrowed_client, fetch_client_kwargs = _build_async_prefetch_client_kwargs(
            model_instance, raw_http_pre, call_timeout=call_to_pre
        )
        if borrowed_client is not None:
            explicit_ft_a = _resolve_media_fetch_timeout(
                model_instance,
                _coerce_http_options(raw_http_pre),
                call_to_pre,
            )
            enforce_redir_a = not bool(
                getattr(borrowed_client, "follow_redirects", False)
            )
            fetched_pairs = await asyncio.gather(
                *(
                    _afetch_external_url(
                        u,
                        borrowed_client,
                        image_loader,
                        request_timeout=explicit_ft_a,
                        enforce_follow_redirects=enforce_redir_a,
                    )
                    for u in urls_to_fetch
                )
            )
            prefetched = dict(fetched_pairs)
        else:
            async with _create_httpx_client(
                fetch_client_kwargs, is_async=True
            ) as http_client:
                fetched_pairs = await asyncio.gather(
                    *(
                        _afetch_external_url(u, http_client, image_loader)
                        for u in urls_to_fetch
                    )
                )
                prefetched = dict(fetched_pairs)

    merged_kw = dict(kwargs)
    merged_kw["_is_async"] = True
    if prefetched:
        merged_kw["_prefetched_urls"] = prefetched
    return _prepare_interactions_request(
        model_instance,
        messages,
        stop=stop,
        tools=tools,
        functions=functions,
        safety_settings=safety_settings,
        tool_config=tool_config,
        generation_config=generation_config,
        cached_content=cached_content,
        tool_choice=tool_choice,
        stream=stream,
        **merged_kw,
    )


__all__ = [
    "_SKIP_THOUGHT_SIGNATURE_B64",
    "_InteractionsPreparedRequest",
    "_aprepare_interactions_request",
    "_build_interactions_generation_config",
    "_convert_block_to_interaction_content",
    "_format_interactions_agent_config",
    "_format_interactions_response_format",
    "_format_interactions_safety_settings",
    "_format_interactions_speech_config",
    "_format_interactions_tool_choice",
    "_format_interactions_tools",
    "_get_last_interaction_messages",
    "_messages_to_interaction_steps",
    "_normalize_signature_b64",
    "_prepare_interactions_request",
    "_resolve_interactions_api_version",
    "_resolve_interactions_thinking",
    "_resolve_store",
]
