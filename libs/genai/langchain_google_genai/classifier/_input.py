"""Convert `GoogleGenAIClassifier` requests to the Decisions API format."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)

from langchain_google_genai.classifier.types import Box, Choice, Noul

if TYPE_CHECKING:
    from pydantic import JsonValue

    from langchain_google_genai.classifier.types import Question, State

_MEDIA_TYPES = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "file": "document",
}
"""LangChain standard media block types mapped to Decisions API input types."""


def to_decision_input(state: State) -> str | list[dict[str, Any]]:
    """Convert state to a string or a list of Decisions API content blocks.

    Strings are sent as-is. Any other state is serialized to JSON text, with
    LangChain standard media blocks sent as native media blocks at the position where
    they appear in the text.

    Args:
        state: Text, messages, or structured state to classify.

    Returns:
        A value suitable for the Decisions API `input` field.

    Raises:
        ValueError: If a media block has neither `base64` nor `url` data.
        TypeError: If the state cannot be represented as JSON.
    """
    if isinstance(state, str):
        return state
    if state is None or isinstance(state, (int, float, bool)):
        msg = (
            "Classifier state must be a string, object, array, BaseMessage, or "
            "sequence of BaseMessage objects."
        )
        raise TypeError(msg)
    media: list[dict[str, Any]] = []
    token = f"lc-media-{uuid.uuid4().hex}"
    text = json.dumps(_to_json_value(state, media, token), ensure_ascii=False)
    if not media:
        return text
    content: list[dict[str, Any]] = []
    position = 0
    for match in re.finditer(rf"{token}-(\d+)", text):
        if match.start() > position:
            content.append({"type": "text", "text": text[position : match.start()]})
        content.append(media[int(match[1])])
        position = match.end()
    if position < len(text):
        content.append({"type": "text", "text": text[position:]})
    return content


def to_decision_questions(questions: dict[str, Question]) -> dict[str, dict[str, Any]]:
    """Convert validated questions to the Decisions API `questions` field.

    JSON instructions and criteria are sent as JSON text.

    Args:
        questions: Validated questions keyed by answer ID.

    Returns:
        Questions in the Decisions API format, keyed by answer ID.

    Raises:
        ValueError: If a question is a `Box`.
    """
    return {
        name: _question_to_api(name, question) for name, question in questions.items()
    }


def _question_to_api(name: str, question: Question) -> dict[str, Any]:
    if isinstance(question, Box):
        msg = f"Question {name!r}: the Decisions API does not support Box questions."
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "type": question.type,
        "instructions": _to_text(question.instructions),
    }
    if isinstance(question, Noul):
        criteria = question.criteria
        outcomes = {
            "true": criteria.true if criteria else None,
            "false": criteria.false if criteria else None,
        }
        descriptions = {
            f"{outcome}_description": _to_text(description)
            for outcome, description in outcomes.items()
            if description is not None
        }
        if descriptions:
            payload["criteria"] = descriptions
    elif isinstance(question, Choice):
        # The API takes a string per option and treats "" as no description.
        payload["criteria"] = {
            label: _to_text(description) if description is not None else ""
            for label, description in question.criteria.items()
        }
    else:
        payload["criteria"] = [_to_text(level) for level in question.criteria]
    return payload


def _to_text(content: JsonValue) -> str:
    """Return text content as-is and other JSON content as JSON text."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _to_json_value(value: object, media: list[dict[str, Any]], token: str) -> Any:
    if isinstance(value, BaseMessage):
        return convert_to_openai_messages(_with_media_tokens(value, media, token))
    if _is_media_block(value):
        return _media_token(value, media, token)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            msg = "Classifier state object keys must be strings."
            raise TypeError(msg)
        return {key: _to_json_value(item, media, token) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item, media, token) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    msg = f"Unsupported classifier state value: {type(value).__name__}."
    raise TypeError(msg)


def _is_media_block(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("type") in _MEDIA_TYPES
        and ("base64" in value or "url" in value)
    )


def _with_media_tokens(
    message: BaseMessage, media: list[dict[str, Any]], token: str
) -> BaseMessage:
    """Replace media blocks in a message with tokens marking where each belongs."""
    if isinstance(message.content, str):
        return message
    content: list[str | dict[str, Any]] = []
    for block in message.content:
        standard = _standard_block(block) if isinstance(block, dict) else None
        if standard is not None and standard.get("type") in _MEDIA_TYPES:
            content.append(
                {"type": "text", "text": _media_token(standard, media, token)}
            )
        else:
            content.append(block)
    return message.model_copy(update={"content": content})


def _standard_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Return the LangChain standard form of a content block, if it has one."""
    try:
        blocks = HumanMessage(content=[block]).content_blocks
    except (ValueError, KeyError, TypeError):
        return None
    return dict(blocks[0]) if len(blocks) == 1 else None


def _media_token(
    block: Mapping[str, Any], media: list[dict[str, Any]], token: str
) -> str:
    media.append(_to_api_media(block))
    return f"{token}-{len(media) - 1}"


def _to_api_media(block: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a LangChain standard media block to a Decisions API media block."""
    part: dict[str, Any] = {"type": _MEDIA_TYPES[block["type"]]}
    mime_type = block.get("mime_type")
    if block.get("base64"):
        part["data"] = block["base64"]
    elif (url := block.get("url")) and url.startswith("data:"):
        header, _, data = url.partition(",")
        part["data"] = data
        mime_type = mime_type or header.removeprefix("data:").split(";")[0] or None
    elif url:
        part["uri"] = url
    else:
        msg = (
            f"Unsupported {block['type']} block for the Decisions API. Only "
            "`base64` and `url` data are supported."
        )
        raise ValueError(msg)
    if mime_type:
        part["mime_type"] = mime_type
    return part
