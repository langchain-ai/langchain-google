"""Usage metadata helpers for Vertex AI Gemini models."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence, cast

import proto  # type: ignore[import-untyped]
from langchain_core.messages.ai import (
    UsageMetadata,
)


def _sanitize_token_detail_key(raw_key: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", raw_key.strip().lower()).strip("_")
    return sanitized or "unknown"


def _extract_token_detail_counts(
    entries: Sequence[Mapping[str, Any]] | None,
    *,
    prefix: str | None = None,
) -> dict[str, int]:
    if not entries:
        return {}
    detail_counts: dict[str, int] = {}
    for entry in entries:
        raw_key = entry.get("modality") or entry.get("type") or entry.get("name")
        if not raw_key:
            continue
        raw_value = (
            entry.get("token_count")
            or entry.get("tokenCount")
            or entry.get("tokens_count")
            or entry.get("tokensCount")
            or entry.get("count")
        )
        try:
            value_int = int(raw_value or 0)
        except (TypeError, ValueError):
            value_int = 0
        if value_int == 0:
            continue
        key = _sanitize_token_detail_key(str(raw_key))
        if prefix:
            key = f"{prefix}{key}"
        detail_counts[key] = detail_counts.get(key, 0) + value_int
    return detail_counts


def _merge_detail_counts(target: dict[str, int], new_entries: dict[str, int]) -> None:
    for key, value in new_entries.items():
        target[key] = target.get(key, 0) + value


def _usage_proto_to_dict(raw_usage: Any) -> dict[str, Any]:
    if raw_usage is None:
        return {}
    if isinstance(raw_usage, Mapping):
        return dict(raw_usage)
    try:
        return proto.Message.to_dict(raw_usage)
    except Exception:  # pragma: no cover
        try:
            return dict(raw_usage)
        except Exception:  # pragma: no cover
            return {}


def coerce_usage_metadata(raw_usage: Any) -> Optional[UsageMetadata]:
    usage_dict = _usage_proto_to_dict(raw_usage)
    if not usage_dict:
        return None

    def _get_int(name: str) -> int:
        value = usage_dict.get(name)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    prompt_tokens = _get_int("prompt_token_count")
    response_tokens = (
        _get_int("candidates_token_count")
        or _get_int("response_token_count")
        or _get_int("output_token_count")
    )
    tool_prompt_tokens = _get_int("tool_use_prompt_token_count")
    reasoning_tokens = _get_int("thoughts_token_count") or _get_int(
        "reasoning_token_count"
    )
    cache_read_tokens = _get_int("cached_content_token_count")

    if all(
        count == 0
        for count in (
            prompt_tokens,
            response_tokens,
            tool_prompt_tokens,
            reasoning_tokens,
            cache_read_tokens,
        )
    ):
        return None

    input_tokens = prompt_tokens + tool_prompt_tokens
    output_tokens = response_tokens + reasoning_tokens
    total_tokens = _get_int("total_token_count") or _get_int("total_tokens")
    if total_tokens == 0 or total_tokens != input_tokens + output_tokens:
        total_tokens = input_tokens + output_tokens

    input_details: dict[str, int] = {}
    if cache_read_tokens:
        input_details["cache_read"] = cache_read_tokens
    if tool_prompt_tokens:
        input_details["tool_use_prompt"] = tool_prompt_tokens
    _merge_detail_counts(
        input_details,
        _extract_token_detail_counts(
            usage_dict.get("prompt_tokens_details")
            or usage_dict.get("promptTokensDetails"),
        ),
    )
    _merge_detail_counts(
        input_details,
        _extract_token_detail_counts(
            usage_dict.get("tool_use_prompt_tokens_details")
            or usage_dict.get("toolUsePromptTokensDetails"),
            prefix="tool_use_prompt_",
        ),
    )
    _merge_detail_counts(
        input_details,
        _extract_token_detail_counts(
            usage_dict.get("cache_tokens_details")
            or usage_dict.get("cacheTokensDetails"),
            prefix="cache_",
        ),
    )

    output_details: dict[str, int] = {}
    if reasoning_tokens:
        output_details["reasoning"] = reasoning_tokens
    for key in (
        "candidates_tokens_details",
        "candidatesTokensDetails",
        "response_tokens_details",
        "responseTokensDetails",
        "output_tokens_details",
        "outputTokensDetails",
        "total_tokens_details",
        "totalTokensDetails",
    ):
        _merge_detail_counts(
            output_details, _extract_token_detail_counts(usage_dict.get(key))
        )

    # Normalize alternate reasoning keys if the API provides e.g. "thought" buckets.
    for alt_key in ("thought", "thoughts", "reasoning_tokens"):
        if alt_key in output_details:
            output_details["reasoning"] = output_details.get(
                "reasoning", 0
            ) + output_details.pop(alt_key)

    payload: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if input_details:
        payload["input_token_details"] = input_details
    if output_details:
        payload["output_token_details"] = output_details

    return cast(UsageMetadata, payload)


def diff_usage_metadata(
    current: Optional[UsageMetadata], previous: Optional[UsageMetadata]
) -> Optional[UsageMetadata]:
    if not current:
        return None
    if not previous:
        return current

    input_delta = current.get("input_tokens", 0) - previous.get("input_tokens", 0)
    output_delta = current.get("output_tokens", 0) - previous.get("output_tokens", 0)
    total_delta = current.get("total_tokens", 0) - previous.get("total_tokens", 0)
    expected_total = input_delta + output_delta
    if total_delta != expected_total:
        total_delta = expected_total

    payload: dict[str, Any] = {
        "input_tokens": input_delta,
        "output_tokens": output_delta,
        "total_tokens": total_delta,
    }

    prev_input_details = cast(
        dict[str, int], previous.get("input_token_details", {}) or {}
    )
    curr_input_details = cast(
        dict[str, int], current.get("input_token_details", {}) or {}
    )
    input_detail_delta = {
        key: curr_input_details.get(key, 0) - prev_input_details.get(key, 0)
        for key in set(prev_input_details).union(curr_input_details)
    }
    input_detail_delta = {k: v for k, v in input_detail_delta.items() if v != 0}
    if input_detail_delta:
        payload["input_token_details"] = input_detail_delta

    prev_output_details = cast(
        dict[str, int], previous.get("output_token_details", {}) or {}
    )
    curr_output_details = cast(
        dict[str, int], current.get("output_token_details", {}) or {}
    )
    output_detail_delta = {
        key: curr_output_details.get(key, 0) - prev_output_details.get(key, 0)
        for key in set(prev_output_details).union(curr_output_details)
    }
    output_detail_delta = {k: v for k, v in output_detail_delta.items() if v != 0}
    if output_detail_delta:
        payload["output_token_details"] = output_detail_delta

    return cast(UsageMetadata, payload)
