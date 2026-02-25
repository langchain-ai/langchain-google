"""Wrapper around Google VertexAI chat-based models.

Vertex supports both v1 and v1beta1 endpoints (`endpoint_version` parameter).
"""

from __future__ import annotations  # noqa
import ast
import base64
from functools import cached_property
import json
import mimetypes
import logging
import re
from operator import itemgetter
import uuid
from typing import (
    Any,
    cast,
    Literal,
    TypedDict,
    overload,
)
from collections.abc import Callable, Sequence
from collections.abc import AsyncIterator, Iterator

import proto  # type: ignore[import-untyped]

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
    agenerate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    convert_to_openai_image_block,
    is_data_content_block,
)
from langchain_core.messages import content as lc_content
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import (
    tool_call_chunk,
    tool_call as create_tool_call,
    invalid_tool_call,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import BaseModel, Field, model_validator
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from vertexai.generative_models import (
    Candidate as VertexCandidate,
    Tool as VertexTool,  # TODO: migrate to google-genai since this is deprecated
)
from vertexai.generative_models._generative_models import (
    ToolConfig,  # TODO: migrate to google-genai since this is deprecated
    SafetySettingsType,
    GenerationConfigType,
    GenerationResponse,
    _convert_schema_dict_to_gapic,
)
from vertexai.language_models import (
    InputOutputTextPair,
)
from google.cloud.aiplatform_v1.types import (
    Content as v1Content,
    FunctionCall as v1FunctionCall,
    FunctionCallingConfig as v1FunctionCallingConfig,
    FunctionResponse as v1FunctionResponse,
    GenerateContentRequest as v1GenerateContentRequest,
    GenerationConfig as v1GenerationConfig,
    Part as v1Part,
    SafetySetting as v1SafetySetting,
    Tool as v1Tool,
    ToolConfig as v1ToolConfig,
)
from google.cloud.aiplatform_v1beta1.types import (
    Blob,
    Candidate,
    CodeExecutionResult,
    Part,
    HarmCategory,
    Content,
    ExecutableCode,
    FileData,
    FunctionCall,
    FunctionResponse,
    GenerateContentRequest,
    GenerationConfig,
    SafetySetting,
    Tool as GapicTool,
    ToolConfig as GapicToolConfig,
    VideoMetadata,
)

from langchain_google_vertexai.data._profiles import _PROFILES
from langchain_google_vertexai._base import _VertexAICommon
from langchain_google_vertexai._compat import _convert_from_v1_to_vertex
from langchain_google_vertexai._image_utils import (
    ImageBytesLoader,
    image_bytes_to_b64_string,
)
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
    _format_model_name,
    replace_defs_in_schema,
    _strip_nullable_anyof,
)
from langchain_google_vertexai.functions_utils import (
    _format_tool_config,
    _ToolConfigDict,
    _tool_choice_to_tool_config,
    _ToolChoiceType,
    _ToolsType,
    _format_to_gapic_tool,
    _ToolType,
)
from pydantic import ConfigDict
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, deprecated as typing_deprecated, is_typeddict
from difflib import get_close_matches


logger = logging.getLogger(__name__)


_allowed_params = [
    "temperature",
    "top_k",
    "top_p",
    "response_mime_type",
    "response_schema",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "candidate_count",
    "seed",
    "response_logprobs",
    "logprobs",
    "labels",
    "audio_timestamp",
    "response_modalities",
    "thinking_budget",
    "include_thoughts",
]
_allowed_beta_params = [
    "media_resolution",
]
_allowed_params_prediction_service = [
    "request",
    "timeout",
    "metadata",
    "labels",
    # Allow controlling GAPIC client retries from callers.
    "retry",
]


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY = (
    "__gemini_function_call_thought_signatures__"
)


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _base64_to_bytes(input_str: str) -> bytes:
    return base64.b64decode(input_str.encode("utf-8"))


class _GeminiGenerateContentKwargs(TypedDict):
    generation_config: GenerationConfigType | None
    safety_settings: SafetySettingsType | None
    tools: list[VertexTool] | None
    tool_config: ToolConfig | None


def _parse_chat_history_gemini(
    history: list[BaseMessage],
    imageBytesLoader: ImageBytesLoader,
    perform_literal_eval_on_string_raw_content: bool | None = False,
) -> tuple[Content | None, list[Content]]:
    """Parse LangChain message history into Gemini format.

    !!! warning

        `perform_literal_eval_on_string_raw_content` should only be set to `True` if you
        fully trust the input, as it may execute arbitrary code.

    Args:
        history: List of LangChain messages.
        imageBytesLoader: An `ImageBytesLoader` instance to handle image loading.
        perform_literal_eval_on_string_raw_content: Whether to attempt to parse string
            content as Python literals (e.g., `list` or `dict`).

    Returns:
        Tuple of `(system_instruction, aiplatform_v1beta1 content)`.
    """
    # Case where content was serialized to v1 format
    for idx, message in enumerate(history):
        if (
            isinstance(message, AIMessage)
            and message.response_metadata.get("output_version") == "v1"
        ):
            # Unpack known v1 content to v1beta format for the request
            #
            # Old content types and any previously serialized messages passed back in to
            # history will skip this, but hit and processed in `_convert_to_parts`
            history[idx] = message.model_copy(
                update={
                    "content": _convert_from_v1_to_vertex(
                        cast("list[lc_content.ContentBlock]", message.content),
                        message.response_metadata.get("model_provider"),
                    )
                }
            )

    def _convert_to_prompt(part: str | dict) -> Part | None:
        if isinstance(part, str):
            return Part(text=part)

        if not isinstance(part, dict):
            msg = f"Message's content is expected to be a dict, got {type(part)}!"  # type: ignore[unreachable, unused-ignore]
            raise ValueError(msg)
        if part["type"] == "text":
            if "thought_signature" in part:
                return Part(
                    text=part["text"],
                    thought_signature=_base64_to_bytes(part["thought_signature"]),
                )
            return Part(text=part["text"])
        if part["type"] == "tool_use":
            if part.get("text"):
                return Part(text=part["text"])
            return None
        if part["type"] == "executable_code":
            if "executable_code" not in part or "language" not in part:
                msg = (
                    "Executable code part must have 'code' and 'language' keys, got "
                    f"{part}"
                )
                raise ValueError(msg)
            return Part(
                executable_code=ExecutableCode(
                    language=part["language"], code=part["executable_code"]
                )
            )
        if part["type"] == "code_execution_result":
            if "code_execution_result" not in part or "outcome" not in part:
                msg = (
                    "Code execution result part must have 'code_execution_result' and "
                    f"'outcome' keys, got {part}"
                )
                raise ValueError(msg)
            return Part(
                code_execution_result=CodeExecutionResult(
                    output=part["code_execution_result"], outcome=part["outcome"]
                )
            )

        if is_data_content_block(part):
            # LangChain standard format
            if part["type"] == "image" and "url" in part:
                oai_content_block = convert_to_openai_image_block(part)
                url = oai_content_block["image_url"]["url"]
                return imageBytesLoader.load_gapic_part(url)
            if part.get("source_type") == "url" or "url" in part:
                url = part.get("url")
                if not url:
                    msg = "Data content block must contain 'url'."
                    raise ValueError(msg)
                mime_type = part.get("mime_type")
                if not mime_type:
                    mime_type, _ = mimetypes.guess_type(url)
                return Part(file_data=FileData(file_uri=url, mime_type=mime_type))
            if "base64" in part or part.get("source_type") == "base64":
                key_name = "base64" if "base64" in part else "data"
                bytes_ = base64.b64decode(part[key_name])
            else:
                msg = "source_type must be url or base64."
                raise ValueError(msg)
            inline_data: dict = {"data": bytes_}
            if "mime_type" in part:
                inline_data["mime_type"] = part["mime_type"]

            return Part(inline_data=Blob(**inline_data))

        if part["type"] == "image_url":
            path = part["image_url"]["url"]
            return imageBytesLoader.load_gapic_part(path)

        # Handle media type like LangChain.js
        # https://github.com/langchain-ai/langchainjs/blob/e536593e2585f1dd7b0afc187de4d07cb40689ba/libs/langchain-google-common/src/utils/gemini.ts#L93-L106
        if part["type"] == "media":
            if "mime_type" not in part:
                msg = f"Missing mime_type in media part: {part}"  # type: ignore[unreachable, unused-ignore]
                raise ValueError(msg)
            mime_type = part["mime_type"]
            proto_part = Part()

            if "data" in part:
                proto_part.inline_data = Blob(data=part["data"], mime_type=mime_type)
            elif "file_uri" in part:
                proto_part.file_data = FileData(
                    file_uri=part["file_uri"], mime_type=mime_type
                )
            else:
                msg = f"Media part must have either data or file_uri: {part}"  # type: ignore[unreachable, unused-ignore]
                raise ValueError(msg)

            if "video_metadata" in part:
                metadata = VideoMetadata(part["video_metadata"])
                proto_part.video_metadata = metadata
            return proto_part

        if part["type"] == "thinking":
            return Part(text=part["thinking"], thought=True)

        if part["type"] == "function_call_signature":
            return None

        msg = "Only text, image_url, and media types are supported!"  # type: ignore[unreachable, unused-ignore]
        raise ValueError(msg)

    def _convert_to_parts(message: BaseMessage) -> list[Part]:
        """Parse LangChain message content into Google parts.

        Used when preparing Human, System and AI messages for sending to the API.

        Handles both legacy (pre-v1) dict-based content blocks and v1 ContentBlock
        objects.
        """
        raw_content = message.content

        # If a user sends a multimodal request with agents, then the full input
        # will be sent as a string due to the ChatPromptTemplate formatting.
        # Because of this, we need to first try to convert the string to its
        # native type (such as list or dict) so that results can be properly
        # appended to the prompt, otherwise they will all be parsed as Text
        # rather than `inline_data`.
        if perform_literal_eval_on_string_raw_content and isinstance(raw_content, str):
            try:
                raw_content = ast.literal_eval(raw_content)
            except SyntaxError:
                pass
            except ValueError:
                pass
        if isinstance(raw_content, (int, float, str)):
            raw_content = [str(raw_content)]
        elif isinstance(raw_content, list):
            # Preserve dict structure when literal_eval successfully parsed the content
            raw_content = [
                item if isinstance(item, dict) else str(item) for item in raw_content
            ]
        else:
            msg = f"Unsupported type: {type(raw_content)}"  # type: ignore[unreachable]
            raise TypeError(msg)
        result = []
        for raw_part in raw_content:
            part = _convert_to_prompt(raw_part)
            if part:
                result.append(part)
        return result

    vertex_messages: list[Content] = []
    system_parts: list[Part] | None = None
    system_instruction = None

    # the last AI Message before a sequence of tool calls
    prev_ai_message: AIMessage | None = None

    for i, message in enumerate(history):
        if isinstance(message, SystemMessage):
            prev_ai_message = None
            system_parts = _convert_to_parts(message)
            if system_instruction is not None:
                system_instruction.parts.extend(system_parts)
            else:
                system_instruction = Content(role="system", parts=system_parts)
            system_parts = None
        elif isinstance(message, HumanMessage):
            prev_ai_message = None
            role = "user"
            parts = _convert_to_parts(message)
            if system_parts is not None:
                parts = system_parts + parts
                system_parts = None
            if vertex_messages and vertex_messages[-1].role == "user":
                prev_parts = list(vertex_messages[-1].parts)
                vertex_messages[-1] = Content(role=role, parts=prev_parts + parts)
            else:
                vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, AIMessage):
            prev_ai_message = message
            role = "model"

            # Previous blocked messages will have empty content which should be ignored
            if not message.content and message.response_metadata.get(
                "is_blocked", False
            ):
                logger.warning("Ignoring blocked AIMessage with empty content")
                continue

            # Extract any function_call_signature blocks from content
            function_call_sigs: dict[int, bytes] = {}
            if isinstance(message.content, list):
                for idx, item in enumerate(message.content):
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function_call_signature"
                    ):
                        sig_str = item.get("signature", "")
                        if sig_str and isinstance(sig_str, str):
                            sig_bytes = _base64_to_bytes(sig_str)
                            if "index" in item:
                                function_call_sigs[item["index"]] = sig_bytes
                            else:
                                function_call_sigs[idx] = sig_bytes

            parts = []
            if message.content:
                parts = _convert_to_parts(message)

            for i, tc in enumerate(message.tool_calls):
                thought_signature: bytes | None = None
                if tool_call_id := tc.get("id"):
                    if tool_call_id in message.additional_kwargs.get(
                        _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
                    ):
                        thought_signature = message.additional_kwargs[
                            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY
                        ][tool_call_id]
                        if isinstance(thought_signature, str):
                            thought_signature = _base64_to_bytes(thought_signature)

                if thought_signature is None:
                    thought_signature = function_call_sigs.get(i)

                function_call = FunctionCall({"name": tc["name"], "args": tc["args"]})
                parts.append(
                    Part(
                        function_call=function_call,
                        **(
                            {"thought_signature": thought_signature}
                            if thought_signature
                            else {}
                        ),
                    )
                )

            if vertex_messages:
                prev_content = vertex_messages[-1]
                prev_content_is_model = prev_content and prev_content.role == "model"
                if prev_content_is_model:
                    prev_parts = list(prev_content.parts)
                    prev_parts.extend(parts)
                    vertex_messages[-1] = Content(role=role, parts=prev_parts)
                    continue

            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, FunctionMessage):
            prev_ai_message = None
            role = "user"

            part = Part(
                function_response=FunctionResponse(
                    name=message.name, response={"content": message.content}
                )
            )
            parts = [part]
            if vertex_messages:
                prev_content = vertex_messages[-1]
                prev_content_is_function = prev_content and prev_content.role == "user"
                if prev_content_is_function:
                    prev_parts = list(prev_content.parts)
                    prev_parts.extend(parts)
                    # replacing last message
                    vertex_messages[-1] = Content(role=role, parts=prev_parts)
                    continue

            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, ToolMessage):
            role = "user"

            # message.name can be null for ToolMessage
            name = message.name
            if name is None and prev_ai_message:
                tool_call_id = message.tool_call_id
                tool_call: ToolCall | None = next(
                    (t for t in prev_ai_message.tool_calls if t["id"] == tool_call_id),
                    None,
                )

                if tool_call is None:
                    msg = (
                        "Message name is empty and can't find"
                        f"corresponding tool call for id: '${tool_call_id}'"
                    )
                    raise ValueError(msg)
                name = tool_call["name"]

            def _parse_content(raw_content: str | dict[Any, Any]) -> dict[Any, Any]:
                if isinstance(raw_content, dict):
                    return raw_content
                if isinstance(raw_content, str):
                    try:
                        content = json.loads(raw_content)
                        # json.loads("2") returns 2 since it's a valid json
                        if isinstance(content, dict):
                            return content
                    except json.JSONDecodeError:
                        pass
                return {"content": raw_content}

            if isinstance(message.content, list):
                parsed_content = [_parse_content(c) for c in message.content]
                if len(parsed_content) > 1:
                    merged_content: dict[Any, Any] = {}
                    for content_piece in parsed_content:
                        for key, value in content_piece.items():
                            if key not in merged_content:
                                merged_content[key] = []
                            merged_content[key].append(value)
                    logger.warning(
                        "Expected content to be a str, got a list with > 1 element."
                        "Merging values together"
                    )
                    content = {k: "".join(v) for k, v in merged_content.items()}
                elif len(parsed_content) == 1:
                    content = parsed_content[0]
                else:
                    content = {"content": ""}
            else:
                content = _parse_content(message.content)

            part = Part(
                function_response=FunctionResponse(
                    name=name,
                    response=content,
                )
            )
            parts = [part]

            prev_content = vertex_messages[-1]
            prev_content_is_tool_response = prev_content and prev_content.role == "user"

            if prev_content_is_tool_response:
                prev_parts = list(prev_content.parts)
                prev_parts.extend(parts)
                # replacing last message
                vertex_messages[-1] = Content(role=role, parts=prev_parts)
                continue

            vertex_messages.append(Content(role=role, parts=parts))
        else:
            msg = f"Unexpected message with type {type(message)} at the position {i}."
            raise ValueError(msg)
    return system_instruction, vertex_messages


def _parse_examples(examples: list[BaseMessage]) -> list[InputOutputTextPair]:
    if len(examples) % 2 != 0:
        msg = (
            f"Expect examples to have an even amount of messages, got {len(examples)}."
        )
        raise ValueError(msg)
    example_pairs = []
    input_text = None
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if not isinstance(example, HumanMessage):
                msg = (
                    f"Expected the first message in a part to be from human, got "
                    f"{type(example)} for the {i}th message."
                )
                raise ValueError(msg)
            # InputOutputTextPair only accepts strings
            input_text = cast("str", example.content)
        if i % 2 == 1:
            if not isinstance(example, AIMessage):
                msg = (
                    f"Expected the second message in a part to be from AI, got "
                    f"{type(example)} for the {i}th message."
                )
                raise ValueError(msg)
            output_text = cast("str", example.content)
            # input_text should always be set by the previous iteration (i % 2 == 0)
            assert input_text is not None, "input_text should be set before output_text"
            pair = InputOutputTextPair(input_text=input_text, output_text=output_text)
            example_pairs.append(pair)
    return example_pairs


def _get_question(messages: list[BaseMessage]) -> HumanMessage:
    """Get The `HumanMessage` at the end of a list of input messages to a chat model."""
    if not messages:
        msg = "You should provide at least one message to start the chat!"
        raise ValueError(msg)
    question = messages[-1]
    if not isinstance(question, HumanMessage):
        msg = f"Last message in the list should be from human, got {question.type}."
        raise ValueError(msg)
    return question


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
    msg = f"Unexpected content type: {type(current_content)}"  # type: ignore[unreachable]
    raise TypeError(msg)


def _collapse_text_content(content: list[Any]) -> str | list[Any]:
    """Collapse list content into a string when it only contains plain text."""
    if not content:
        return ""
    if all(isinstance(item, str) for item in content):
        return "".join(content)
    if all(
        isinstance(item, dict)
        and item.get("type") == "text"
        and set(item.keys()).issubset({"type", "text"})
        for item in content
    ):
        return "".join(item.get("text", "") for item in content)
    return content


@overload
def _parse_response_candidate(
    response_candidate: Candidate | VertexCandidate,
    streaming: Literal[False] = False,
) -> AIMessage: ...


@overload
def _parse_response_candidate(
    response_candidate: Candidate | VertexCandidate, streaming: Literal[True]
) -> AIMessageChunk: ...


def _parse_response_candidate(
    response_candidate: Candidate | VertexCandidate, streaming: bool = False
) -> AIMessage:
    content: None | str | list[str | dict[str, Any]] = None
    additional_kwargs = {}
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    for part in response_candidate.content.parts:
        text: str | None = part.text
        try:
            if hasattr(part, "text") and part.text is not None:
                text = part.text
        except AttributeError:
            pass

        if hasattr(part, "thought") and part.thought:
            thinking_message = {
                "type": "thinking",
                "thinking": part.text,
            }
            content = _append_to_content(content, thinking_message)
        elif text is not None and text:
            if hasattr(part, "thought_signature") and part.thought_signature:
                text_message = {
                    "type": "text",
                    "text": text,
                    "thought_signature": _bytes_to_base64(part.thought_signature),
                }
                content = _append_to_content(content, text_message)
            else:
                content = _append_to_content(content, text)

        if part.function_call:
            # For backward compatibility we store a function call in additional_kwargs,
            # but in general the full set of function calls is stored in tool_calls.
            function_call = {"name": part.function_call.name}
            # dump to match other function calling llm for now
            function_call_args_dict = proto.Message.to_dict(part.function_call)["args"]
            function_call["arguments"] = json.dumps(
                {k: function_call_args_dict[k] for k in function_call_args_dict}
            )
            additional_kwargs["function_call"] = function_call

            tool_call_id = function_call.get("id", str(uuid.uuid4()))
            if streaming:
                index = function_call.get("index")
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=tool_call_id,
                        index=int(index) if index else None,
                    )
                )
            else:
                try:
                    tool_calls_dicts = parse_tool_calls(
                        [{"function": function_call}],
                        return_id=False,
                    )
                    tool_calls.extend(
                        [
                            create_tool_call(
                                name=tool_call["name"],
                                args=tool_call["args"],
                                id=tool_call.get("id", tool_call_id),
                            )
                            for tool_call in tool_calls_dicts
                        ]
                    )
                except Exception as e:
                    invalid_tool_calls.append(
                        invalid_tool_call(
                            name=function_call.get("name"),
                            args=function_call.get("arguments"),
                            id=tool_call_id,
                            error=str(e),
                        )
                    )

            if getattr(part, "thought_signature", None):
                # store dict of {tool_call_id: thought_signature}
                if hasattr(part, "thought_signature") and isinstance(
                    part.thought_signature, bytes
                ):
                    thought_signature = _bytes_to_base64(part.thought_signature)
                    if (
                        _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY
                        not in additional_kwargs
                    ):
                        additional_kwargs[
                            _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY
                        ] = {}
                    additional_kwargs[_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY][
                        tool_call_id
                    ] = thought_signature

        if hasattr(part, "executable_code") and part.executable_code is not None:
            if part.executable_code.code and part.executable_code.language:
                code_message = {
                    "type": "executable_code",
                    "executable_code": part.executable_code.code,
                    "language": part.executable_code.language,
                }
                content = _append_to_content(content, code_message)

        if (
            (
                hasattr(part, "code_execution_result")
                and part.code_execution_result is not None
            )
            and part.code_execution_result.output
            and part.code_execution_result.outcome
        ):
            execution_result = {
                "type": "code_execution_result",
                # Name output -> code_execution_result for consistency with
                # langchain-google-genai
                "code_execution_result": part.code_execution_result.output,
                "outcome": part.code_execution_result.outcome,
            }
            content = _append_to_content(content, execution_result)

        if part.inline_data.mime_type.startswith("image/"):
            image_format = part.inline_data.mime_type[6:]
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": image_bytes_to_b64_string(
                        part.inline_data.data, image_format=image_format
                    )
                },
            }
            content = _append_to_content(content, image_message)

    if content is None:
        content = ""
    if isinstance(content, list):
        content = _collapse_text_content(content)

    if streaming:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        additional_kwargs=additional_kwargs,
        invalid_tool_calls=invalid_tool_calls,
    )


def _completion_with_retry(
    generation_method: Callable,
    *,
    max_retries: int,
    run_manager: CallbackManagerForLLMRun | None = None,
    wait_exponential_kwargs: dict[str, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries,
        run_manager=run_manager,
        wait_exponential_kwargs=wait_exponential_kwargs,
    )

    @retry_decorator
    def _completion_with_retry_inner(generation_method: Callable, **kwargs: Any) -> Any:
        return generation_method(**kwargs)

    # If user requested 0 retries, disable GAPIC retries too unless explicitly set.
    if max_retries <= 0 and "retry" not in kwargs:
        kwargs["retry"] = None

    params = {
        k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service
    }
    return _completion_with_retry_inner(
        generation_method,
        **params,
    )


async def _acompletion_with_retry(
    generation_method: Callable,
    *,
    max_retries: int,
    run_manager: AsyncCallbackManagerForLLMRun | None = None,
    wait_exponential_kwargs: dict[str, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries,
        run_manager=run_manager,
        wait_exponential_kwargs=wait_exponential_kwargs,
    )

    @retry_decorator
    async def _completion_with_retry_inner(
        generation_method: Callable, **kwargs: Any
    ) -> Any:
        return await generation_method(**kwargs)

    # If user requested 0 retries, disable GAPIC retries too unless explicitly set.
    if max_retries <= 0 and "retry" not in kwargs:
        kwargs["retry"] = None

    params = {
        k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service
    }
    return await _completion_with_retry_inner(
        generation_method,
        **params,
    )


@typing_deprecated(
    "Use [`ChatGoogleGenerativeAI`][langchain_google_genai.ChatGoogleGenerativeAI] "
    "instead."
)
@deprecated(
    since="3.2.0",
    removal="4.0.0",
    alternative_import="langchain_google_genai.ChatGoogleGenerativeAI",
)
class ChatVertexAI(_VertexAICommon, BaseChatModel):
    r"""Google Cloud Vertex AI chat model integration.

    Setup:
        You must either:

        - Have credentials configured for your environment (gcloud, workload identity,
            etc...)
        - Store the path to a service account JSON file as the
            `GOOGLE_APPLICATION_CREDENTIALS` environment variable

        This codebase uses the `google.auth` library which first looks for the
        application credentials variable mentioned above, and then looks for
        system-level auth.

        **More information:**

        - [Credential types](https://cloud.google.com/docs/authentication/application-default-credentials#GAC)
        - `google.auth` [API reference](https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth)

    Key init args — completion params:
        model: str
            Name of ChatVertexAI model to use. e.g. `'gemini-2.0-flash-001'`,
            `'gemini-2.5-pro'`, etc.
        temperature: Optional[float]
            Sampling temperature.
        seed: Optional[int]
            Sampling integer to use.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        stop: Optional[List[str]]
            Default stop sequences.
        safety_settings: Optional[Dict[vertexai.generative_models.HarmCategory, vertexai.generative_models.HarmBlockThreshold]]
            The default safety settings to use for all generations.

    Key init args — client params:
        max_retries: int
            Max number of retries.
        wait_exponential_kwargs: Optional[dict[str, float]]
            Optional dictionary with parameters for wait_exponential:
            - multiplier: Initial wait time multiplier (default: `1.0`)
            - min: Minimum wait time in seconds (default: `4.0`)
            - max: Maximum wait time in seconds (default: `10.0`)
            - exp_base: Exponent base to use (default: `2.0`)
        credentials: Optional[google.auth.credentials.Credentials]
            The default custom credentials to use when making API calls. If not
            provided, credentials will be ascertained from the environment.
        project: Optional[str]
            The default GCP project to use when making Vertex API calls.
        location: str = "us-central1"
            The default location to use when making API calls.
        request_parallelism: int = 5
            The amount of parallelism allowed for requests issued to VertexAI models.
        base_url: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            max_retries=6,
            stop=None,
            # other params...
        )
        ```

    Thinking:
        For thinking models, you have the option to adjust the number of internal
        thinking tokens used (`thinking_budget`) or to disable thinking altogether.
        Note that not all models allow disabling thinking.

        See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for
        more details on thinking models.

        To see a thinking model's thoughts, set `include_thoughts=True` to have the
        model's reasoning summaries included in the response.

        ```python
        llm = ChatVertexAI(
            model="gemini-2.5-flash",
            include_thoughts=True,
        )
        ai_msg = llm.invoke("How many 'r's are in the word 'strawberry'?")
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        llm.invoke(messages)
        ```

        ```python
        AIMessage(
            content="J'adore programmer. ",
            response_metadata={
                "is_blocked": False,
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                ],
                "citation_metadata": None,
                "usage_metadata": {
                    "prompt_token_count": 17,
                    "candidates_token_count": 7,
                    "total_token_count": 24,
                },
            },
            id="run-925ce305-2268-44c4-875f-dde9128520ad-0",
        )
        ```

    Stream:
        ```python
        for chunk in llm.stream(messages):
            print(chunk)
        ```

        ```python
        AIMessageChunk(
            content="J",
            response_metadata={
                "is_blocked": False,
                "safety_ratings": [],
                "citation_metadata": None,
            },
            id="run-9df01d73-84d9-42db-9d6b-b1466a019e89",
        )
        AIMessageChunk(
            content="'adore programmer. ",
            response_metadata={
                "is_blocked": False,
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                ],
                "citation_metadata": None,
            },
            id="run-9df01d73-84d9-42db-9d6b-b1466a019e89",
        )
        AIMessageChunk(
            content="",
            response_metadata={
                "is_blocked": False,
                "safety_ratings": [],
                "citation_metadata": None,
                "usage_metadata": {
                    "prompt_token_count": 17,
                    "candidates_token_count": 7,
                    "total_token_count": 24,
                },
            },
            id="run-9df01d73-84d9-42db-9d6b-b1466a019e89",
        )
        ```

        ```python
        stream = llm.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

        ```python
        AIMessageChunk(
            content="J'adore programmer. ",
            response_metadata={
                "is_blocked": False,
                "safety_ratings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability_label": "NEGLIGIBLE",
                        "probability_score": 0.1,
                        "blocked": False,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severity_score": 0.1,
                    },
                ],
                "citation_metadata": None,
                "usage_metadata": {
                    "prompt_token_count": 17,
                    "candidates_token_count": 7,
                    "total_token_count": 24,
                },
            },
            id="run-b7f7492c-4cb5-42d0-8fc3-dce9b293b0fb",
        )
        ```

    Async invocation:
        ```python
        await llm.ainvoke(messages)

        # stream
        async for chunk in (await llm.astream(messages))

        # batch
        await llm.abatch([messages])
        ```

    Context Caching:
        Context caching allows you to store and reuse content (e.g., PDFs, images) for
        faster processing.

        The `cached_content` parameter accepts a cache name created via the Google
        Generative AI API with Vertex AI.

        !!! example "Content caching"
            This caches content from GCS and queries it.

            ```python
            from google import genai
            from google.genai.types import (
                Content,
                CreateCachedContentConfig,
                HttpOptions,
                Part,
            )
            from langchain_google_vertexai import ChatVertexAI
            from langchain_core.messages import HumanMessage

            client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))

            contents = [
                Content(
                    role="user",
                    parts=[
                        Part.from_uri(
                            file_uri="gs://your-bucket/file1",
                            mime_type="application/pdf",
                        ),
                        Part.from_uri(
                            file_uri="gs://your-bucket/file2",
                            mime_type="image/jpeg",
                        ),
                    ],
                )
            ]

            cache = client.caches.create(
                model="gemini-2.5-flash",
                config=CreateCachedContentConfig(
                    contents=contents,
                    system_instruction="You are an expert content analyzer.",
                    display_name="content-cache",
                    ttl="300s",
                ),
            )

            llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                cached_content=cache.name,
            )
            message = HumanMessage(
                content="Provide a summary of the key information across the content."
            )
            llm.invoke([message])
            ```

    Tool calling:
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
                "id": "2a2401fa-40db-470d-83ce-4e52de910d9e",
            },
            {
                "name": "GetWeather",
                "args": {"location": "New York City, NY"},
                "id": "96761deb-ab7f-4ef9-b4b4-6d44562fc46e",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "Los Angeles, CA"},
                "id": "9147d532-abee-43a2-adb5-12f164300484",
            },
            {
                "name": "GetPopulation",
                "args": {"location": "New York City, NY"},
                "id": "c43374ea-bde5-49ca-8487-5b83ebeea1e6",
            },
        ]
        ```

        See `bind_tools` for more.

    Built-in search:
        ```python
        from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(model="gemini-2.5-flash")
        resp = llm.invoke(
            "When is the next total solar eclipse in US?",
            tools=[VertexTool(google_search={})],
        )
        ```

    Built-in code execution:
        ```python
        from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
        from langchain_google_vertexai import ChatVertexAI

        llm = ChatVertexAI(model="gemini-2.5-flash")
        resp = llm.invoke(
            "What is 3^3?",
            tools=[VertexTool(code_execution={})],
        )
        ```

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: Optional[int] = Field(
                default=None, description="How funny the joke is, from 1 to 10"
            )


        structured_llm = llm.with_structured_output(Joke)
        structured_llm.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="What do you call a cat that loves to bowl?",
            punchline="An alley cat!",
            rating=None,
        )
        ```

        See `with_structured_output` for more.

    Image input:
        ```python
        import base64
        import httpx
        from langchain_core.messages import HumanMessage

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        The weather in this image appears to be sunny and pleasant. The sky is a bright
        blue with scattered white clouds, suggesting a clear and mild day. The lush
        green grass indicates recent rainfall or sufficient moisture. The absence of...
        ```

        You can also point to GCS files which is faster / more efficient because bytes are transferred back and forth.

        ```python
        llm.invoke(
            [
                HumanMessage(
                    [
                        "What's in the image?",
                        {
                            "type": "media",
                            "file_uri": "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                            "mime_type": "image/jpeg",
                        },
                    ]
                )
            ]
        ).content
        ```

        ```txt
        The image is of five blueberry scones arranged on a piece of baking paper. Here
        is a list of what is in the picture:* **Five blueberry scones:** They are
        scattered across the parchment paper, dusted with powdered sugar.  * **Two...
        ```

    PDF input:
        ```python
        import base64
        from langchain_core.messages import HumanMessage

        pdf_bytes = open("/path/to/your/test.pdf", "rb").read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the document in a sentence"},
                {
                    "type": "file",
                    "mime_type": "application/pdf",
                    "base64": pdf_base64,
                },
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        This research paper describes a system developed for SemEval-2025 Task 9, which
        aims to automate the detection of food hazards from recall reports, addressing
        the class imbalance problem by leveraging LLM-based data augmentation...
        ```

        You can also point to GCS files.

        ```python
        llm.invoke(
            [
                HumanMessage(
                    [
                        "describe the document in a sentence",
                        {
                            "type": "media",
                            "file_uri": "gs://cloud-samples-data/generative-ai/pdf/1706.03762v7.pdf",
                            "mime_type": "application/pdf",
                        },
                    ]
                )
            ]
        ).content
        ```

        ```txt
        The article introduces Transformer, a new model architecture for sequence
        transduction based solely on attention mechanisms, outperforming previous models
        in machine translation tasks and demonstrating good generalization to English...
        ```

    Video input:
        ```python
        import base64
        from langchain_core.messages import HumanMessage

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
                    "mime_type": "video/mp4",
                    "base64": video_base64,
                },
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        Tom and Jerry, along with a turkey, engage in a chaotic Thanksgiving-themed
        adventure involving a corn-on-the-cob chase, maze antics, and a disastrous
        attempt to prepare a turkey dinner.
        ```

        You can also pass YouTube URLs directly:

        ```python
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "text", "text": "summarize the video in 3 sentences."},
                {
                    "type": "media",
                    "file_uri": "https://www.youtube.com/watch?v=9hE5-98ZeCg",
                    "mime_type": "video/mp4",
                },
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        The video is a demo of multimodal live streaming in Gemini 2.0. The narrator is
        sharing his screen in AI Studio and asks if the AI can see it. The AI then reads
        text that is highlighted on the screen, defines the word “multimodal,” and...
        ```

        You can also point to GCS files.

        ```python
        llm = ChatVertexAI(model="gemini-2.5-pro")

        llm.invoke(
            [
                HumanMessage(
                    [
                        "What's in the video?",
                        {
                            "type": "media",
                            "file_uri": "gs://cloud-samples-data/video/animals.mp4",
                            "mime_type": "video/mp4",
                        },
                    ]
                )
            ]
        ).content
        ```

        ```txt
        The video is about a new feature in Google Photos called "Zoomable Selfies". The
        feature allows users to take selfies with animals at the zoo. The video shows
        several examples of people taking selfies with animals, including a tiger,...
        ```

    Audio input:
        ```python
        import base64
        from langchain_core.messages import HumanMessage

        audio_bytes = open("/path/to/your/audio.mp3", "rb").read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": "summarize this audio in a sentence"},
                {
                    "type": "file",
                    "mime_type": "audio/mp3",
                    "base64": audio_base64,
                },
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```python
        "In this episode of the Made by Google podcast, Stephen Johnson and Simon Tokumine discuss NotebookLM, a tool designed to help users understand complex material in various modalities, with a focus on its unexpected uses, the development of audio overviews, and the implementation of new features like mind maps and source discovery."
        ```

        You can also point to GCS files.

        ```python
        from langchain_core.messages import HumanMessage

        llm = ChatVertexAI(model="gemini-2.5-flash")

        llm.invoke(
            [
                HumanMessage(
                    [
                        "What's this audio about?",
                        {
                            "type": "media",
                            "file_uri": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
                            "mime_type": "audio/mpeg",
                        },
                    ]
                )
            ]
        ).content
        ```

        ```python
        "This audio is an interview with two product managers from Google who work on Pixel feature drops. They discuss how feature drops are important for showcasing how Google devices are constantly improving and getting better. They also discuss some of the highlights of the January feature drop and the new features coming in the March drop for Pixel phones and Pixel watches. The interview concludes with discussion of how user feedback is extremely important to them in deciding which features to include in the feature drops."
        ```

    Token usage:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 17, "output_tokens": 7, "total_tokens": 24}
        ```

    Logprobs:
        ```python
        llm = ChatVertexAI(model="gemini-2.5-flash", logprobs=True)
        ai_msg = llm.invoke(messages)
        ai_msg.response_metadata["logprobs_result"]
        ```

        ```python
        [
            {"token": "J", "logprob": -1.549651415189146e-06, "top_logprobs": []},
            {"token": "'", "logprob": -1.549651415189146e-06, "top_logprobs": []},
            {"token": "adore", "logprob": 0.0, "top_logprobs": []},
            {
                "token": " programmer",
                "logprob": -1.1922384146600962e-07,
                "top_logprobs": [],
            },
            {"token": ".", "logprob": -4.827636439586058e-05, "top_logprobs": []},
            {"token": " ", "logprob": -0.018011733889579773, "top_logprobs": []},
            {"token": "\\n", "logprob": -0.0008687592926435173, "top_logprobs": []},
        ]
        ```

    Response metadata:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
            "is_blocked": False,
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
            ],
            "usage_metadata": {
                "prompt_token_count": 17,
                "candidates_token_count": 7,
                "total_token_count": 24,
            },
        }
        ```

    Safety settings:
        ```python
        from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

        llm = ChatVertexAI(
            model="gemini-2.5-pro",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )

        llm.invoke(messages).response_metadata
        ```

        ```python
        {
            "is_blocked": False,
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "probability_label": "NEGLIGIBLE",
                    "probability_score": 0.1,
                    "blocked": False,
                    "severity": "HARM_SEVERITY_NEGLIGIBLE",
                    "severity_score": 0.1,
                },
            ],
            "usage_metadata": {
                "prompt_token_count": 17,
                "candidates_token_count": 7,
                "total_token_count": 24,
            },
        }
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    "Underlying model name."

    examples: list[BaseMessage] | None = None

    response_mime_type: str | None = None
    """Output response MIME type of the generated candidate text.

    Supported MIME types:

    * `'text/plain'`: (default) Text output.
    * `'application/json'`: JSON response in the candidates.
    * `'text/x.enum'`: Enum in plain text.

    The model also needs to be prompted to output the appropriate response type,
    otherwise the behavior is undefined.

    This is a preview feature.
    """

    response_schema: dict[str, Any] | None = None
    """Enforce a schema to the output.

    The format of the dictionary should follow Open API schema.
    """

    cached_content: str | None = None
    """Whether to use the model in cache mode.

    Must be a string containing the cache name (A sequence of numbers)
    """

    logprobs: bool | int = False
    """Whether to return logprobs as part of `AIMessage.response_metadata`.

    If `False`, don't return logprobs. If `True`, return logprobs for top candidate.
    If `int`, return logprobs for top `logprobs` candidates.
    """

    labels: dict[str, str] | None = None
    """Optional tag llm calls with metadata to help in tracebility and biling."""

    perform_literal_eval_on_string_raw_content: bool = False
    """Whether to perform literal eval on string raw content."""

    wait_exponential_kwargs: dict[str, float] | None = None
    """Optional dictionary with parameters for `wait_exponential`:

    - `multiplier`: Initial wait time multiplier (Default: `1.0`)
    - `min`: Minimum wait time in seconds (Default: `4.0`)
    - `max`: Maximum wait time in seconds (Default: `10.0`)
    - `exp_base`: Exponent base to use (Default: `2.0`)
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any unexpected initialization parameters."""

    def __init__(self, *, model_name: str | None = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize `model_name` as a valid arg and for arg
        validation.
        """
        if model_name:
            kwargs["model_name"] = model_name

        # Get all valid field names, including aliases
        valid_fields = set()
        for field_name, field_info in type(self).model_fields.items():
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
                    f"Unexpected argument '{arg}' provided to ChatVertexAI.{suggestion}"
                )
        super().__init__(**kwargs)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            `["langchain", "chat_models", "vertexai"]`
        """
        return ["langchain", "chat_models", "vertexai"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if self.labels:
            for key, value in self.labels.items():
                if not re.match(r"^[a-z][a-z0-9-_]{0,62}$", key):
                    msg = f"Invalid label key: {key}"
                    raise ValueError(msg)
                if value and len(value) > 63:
                    msg = f"Label value too long: {value}"
                    raise ValueError(msg)
        return self

    @cached_property
    def _image_bytes_loader_client(self):
        return ImageBytesLoader(project=self.project)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the python package exists in environment."""
        if self.full_model_name is not None:
            pass
        elif self.tuned_model_name is not None:
            self.full_model_name = _format_model_name(
                self.tuned_model_name,
                location=self.location,
                project=cast("str", self.project),
            )
        else:
            self.full_model_name = _format_model_name(
                self.model_name,
                location=self.location,
                project=cast("str", self.project),
            )

        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            model_id = re.sub(r"-\d{3}$", "", self.model_name.replace("models/", ""))
            self.profile = _get_default_model_profile(model_id)
        return self

    def _prepare_params(
        self,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        params = super()._prepare_params(stop=stop, stream=stream, **kwargs)

        response_mime_type = kwargs.get("response_mime_type", self.response_mime_type)
        if response_mime_type is not None:
            params["response_mime_type"] = response_mime_type

        response_schema = kwargs.get("response_schema", self.response_schema)
        if response_schema is not None:
            allowed_mime_types = ("application/json", "text/x.enum")
            if response_mime_type not in allowed_mime_types:
                error_message = (
                    "`response_schema` is only supported when "
                    f"`response_mime_type` is set to one of {allowed_mime_types}"
                )
                raise ValueError(error_message)

            gapic_response_schema = _convert_schema_dict_to_gapic(response_schema)
            params["response_schema"] = gapic_response_schema

        audio_timestamp = kwargs.get("audio_timestamp", self.audio_timestamp)
        if audio_timestamp is not None:
            params["audio_timestamp"] = audio_timestamp

        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            params["thinking_config"] = {"thinking_budget": thinking_budget}
        # Remove from top-level params since GenerationConfig expects it nested
        _ = params.pop("thinking_budget", None)

        include_thoughts = kwargs.get("include_thoughts", self.include_thoughts)
        if include_thoughts is not None:
            if "thinking_config" not in params:
                params["thinking_config"] = {}
            params["thinking_config"]["include_thoughts"] = include_thoughts
        # Remove from top-level params since GenerationConfig expects it nested
        _ = params.pop("include_thoughts", None)

        media_resolution = kwargs.get("media_resolution")
        if media_resolution is not None:
            params["media_resolution"] = media_resolution
        response_modalities = kwargs.get(
            "response_modalities", self.response_modalities
        )
        if response_modalities is not None:
            params["response_modalities"] = response_modalities

        return params

    def _get_invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get invocation parameters for tracing."""
        # Get standard parameters from base (_type, model_name, etc.)
        params = super()._get_invocation_params(stop=stop, **kwargs)

        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            params["thinking_budget"] = thinking_budget

        include_thoughts = kwargs.get("include_thoughts", self.include_thoughts)
        if include_thoughts is not None:
            params["include_thoughts"] = include_thoughts

        return params

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="google_vertexai",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop

        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The `CallbackManager` for LLM run. Not used at the moment.
            stream: Whether to use the streaming endpoint.

        Returns:
            The `ChatResult` that contains outputs generated by the model.

        Raises:
            ValueError: If the last message in the list is not from a human.
        """
        if stream is True or (stream is None and self.streaming):
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        return self._generate_gemini(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def _generation_config_gemini(
        self,
        stop: list[str] | None = None,
        stream: bool = False,
        *,
        logprobs: int | bool = False,
        **kwargs: Any,
    ) -> GenerationConfig | v1GenerationConfig:
        """Prepares `GenerationConfig` part of the request.

        [More info](https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#generationconfig)
        """
        if logprobs and isinstance(logprobs, bool):
            kwargs["response_logprobs"] = logprobs
        elif logprobs and isinstance(logprobs, int):
            kwargs["response_logprobs"] = True
            kwargs["logprobs"] = logprobs
        else:
            pass

        if self.endpoint_version == "v1":
            return v1GenerationConfig(
                **self._prepare_params(
                    stop=stop,
                    stream=stream,
                    **{k: v for k, v in kwargs.items() if k in _allowed_params},
                )
            )

        return GenerationConfig(
            **self._prepare_params(
                stop=stop,
                stream=stream,
                **{k: v for k, v in kwargs.items() if k in _allowed_params},
                **{k: v for k, v in kwargs.items() if k in _allowed_beta_params},
            )
        )

    def _safety_settings_gemini(
        self, safety_settings: SafetySettingsType | None
    ) -> Sequence[SafetySetting] | None:
        """Prepares `SafetySetting` part of the request.

        [More info](https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#safetysetting)
        """
        if safety_settings is None:
            if self.safety_settings:
                return self._safety_settings_gemini(self.safety_settings)
            return None
        if isinstance(safety_settings, list):
            # Convert from vertexai SafetySetting to gapic SafetySetting if needed
            return safety_settings  # type: ignore[return-value] # Assuming compatible types
        if isinstance(safety_settings, dict):
            formatted_safety_settings = []
            for category, threshold in safety_settings.items():
                if isinstance(category, str):
                    category = HarmCategory[category]  # type: ignore[misc]
                if isinstance(threshold, str):
                    threshold = SafetySetting.HarmBlockThreshold[threshold]  # type: ignore[misc]

                formatted_safety_settings.append(
                    SafetySetting(
                        category=HarmCategory(category),
                        threshold=SafetySetting.HarmBlockThreshold(threshold),
                    )
                )
            return formatted_safety_settings
        # Should never reach here due to the above conditions
        raise ValueError("safety_settings should be either a list, dict, or None")

    def _prepare_request_gemini(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        stream: bool = False,
        tools: _ToolsType | None = None,
        functions: _ToolsType | None = None,
        tool_config: _ToolConfigDict | ToolConfig | None = None,
        safety_settings: SafetySettingsType | None = None,
        cached_content: str | None = None,
        *,
        tool_choice: _ToolChoiceType | None = None,
        logprobs: int | bool | None = None,
        **kwargs,
    ) -> v1GenerateContentRequest | GenerateContentRequest:
        system_instruction, contents = _parse_chat_history_gemini(
            messages,
            self._image_bytes_loader_client,
            perform_literal_eval_on_string_raw_content=self.perform_literal_eval_on_string_raw_content,
        )
        formatted_tools = self._tools_gemini(tools=tools, functions=functions)
        if tool_config:
            tool_config = self._tool_config_gemini(tool_config=tool_config)
        elif tool_choice:
            all_names = [
                f.name
                for tool in (formatted_tools or [])
                for f in tool.function_declarations
            ]
            tool_config = _tool_choice_to_tool_config(tool_choice, all_names)
        else:
            pass
        formatted_safety_settings = self._safety_settings_gemini(safety_settings)
        logprobs = logprobs if logprobs is not None else self.logprobs
        logprobs = logprobs if isinstance(logprobs, (int, bool)) else False
        generation_config = self._generation_config_gemini(
            stream=stream, stop=stop, logprobs=logprobs, **kwargs
        )

        def _content_to_v1(contents: list[Content]) -> list[v1Content]:
            v1_contens = []
            for content in contents:
                v1_parts = []
                for part in content.parts:
                    raw_part = proto.Message.to_dict(part)
                    _ = raw_part.pop("thought")
                    _ = raw_part.pop("thought_signature", None)

                    if "function_call" in raw_part and isinstance(
                        raw_part["function_call"], dict
                    ):
                        _ = raw_part["function_call"].pop("id", None)
                        v1_parts.append(
                            v1Part(
                                function_call=v1FunctionCall(
                                    **raw_part["function_call"]
                                )
                            )
                        )

                    elif "function_response" in raw_part and isinstance(
                        raw_part["function_response"], dict
                    ):
                        _ = raw_part["function_response"].pop("id", None)
                        v1_parts.append(
                            v1Part(
                                function_response=v1FunctionResponse(
                                    **raw_part["function_response"]
                                )
                            )
                        )

                    else:
                        v1_parts.append(v1Part(**raw_part))
                v1_contens.append(v1Content(role=content.role, parts=v1_parts))
            return v1_contens

        v1_system_instruction, v1_tools, v1_tool_config, v1_safety_settings = (
            None,
            None,
            None,
            None,
        )
        if self.endpoint_version == "v1":
            v1_system_instruction = (
                _content_to_v1([system_instruction])[0] if system_instruction else None
            )
            if formatted_tools:
                v1_tools = [v1Tool(**proto.Message.to_dict(t)) for t in formatted_tools]

            if tool_config:
                v1_tool_config = v1ToolConfig(
                    function_calling_config=v1FunctionCallingConfig(
                        # Assuming gapic type has function_calling_config
                        # mypy can't guarantee all variants of `tool_config` have the
                        # function_calling_config attribute
                        **proto.Message.to_dict(tool_config.function_calling_config)  # type: ignore[union-attr]
                    )
                )

            if formatted_safety_settings:
                v1_safety_settings = [
                    v1SafetySetting(
                        category=s.category,
                        method=s.method,
                        threshold=s.threshold,
                    )
                    for s in formatted_safety_settings
                ]

        if (self.cached_content is not None) or (cached_content is not None):
            selected_cached_content = self.cached_content or cached_content

            full_cache_name = self._request_from_cached_content(
                cached_content=selected_cached_content,  # type: ignore
                system_instruction=system_instruction,
                tools=formatted_tools,
                tool_config=tool_config,
            )

            if self.endpoint_version == "v1":
                return GenerateContentRequest(
                    contents=_content_to_v1(contents),
                    model=self.full_model_name,
                    safety_settings=v1_safety_settings,
                    generation_config=generation_config,
                    cached_content=full_cache_name,
                )

            return GenerateContentRequest(
                contents=contents,
                model=self.full_model_name,
                safety_settings=formatted_safety_settings,
                generation_config=generation_config,
                cached_content=full_cache_name,
            )

        if self.endpoint_version == "v1":
            return v1GenerateContentRequest(
                contents=_content_to_v1(contents),
                system_instruction=v1_system_instruction,
                tools=v1_tools,
                tool_config=v1_tool_config,
                safety_settings=v1_safety_settings,
                generation_config=generation_config,
                model=self.full_model_name,
                labels=self.labels,
            )

        return GenerateContentRequest(
            contents=contents,
            system_instruction=system_instruction,
            tools=formatted_tools,
            tool_config=tool_config,
            safety_settings=formatted_safety_settings,
            generation_config=generation_config,
            model=self.full_model_name,
            labels=self.labels,
        )

    def _request_from_cached_content(
        self,
        cached_content: str,
        system_instruction: Content | None,
        tools: Sequence[GapicTool] | None,
        tool_config: _ToolConfigDict | ToolConfig | None,
    ) -> str:
        not_allowed_parameters = [
            ("system_instructions", system_instruction),
            ("tools", tools),
            ("tool_config", tool_config),
        ]

        for param_name, parameter in not_allowed_parameters:
            if parameter:
                message = (
                    f"Using cached content. Parameter `{param_name}` will be ignored. "
                )
                logger.warning(message)

        return (
            f"projects/{self.project}/locations/{self.location}/"
            f"cachedContents/{cached_content}"
        )

    def _generate_gemini(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)
        timeout = kwargs.pop("timeout", self.timeout)
        response = _completion_with_retry(
            self.prediction_client.generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            timeout=timeout,
            **kwargs,
        )
        return self._gemini_response_to_chat_result(response)

    async def _agenerate_gemini(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        timeout = kwargs.pop("timeout", self.timeout)
        response = await _acompletion_with_retry(
            self.async_prediction_client.generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=self._prepare_request_gemini(
                messages=messages, stop=stop, **kwargs
            ),
            metadata=self.default_metadata,
            timeout=timeout,
            **kwargs,
        )
        return self._gemini_response_to_chat_result(response)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        [More info](https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#counttokensrequest)
        """
        _, contents = _parse_chat_history_gemini(
            [HumanMessage(content=text)],
            self._image_bytes_loader_client,
            perform_literal_eval_on_string_raw_content=self.perform_literal_eval_on_string_raw_content,
        )
        response = self.prediction_client.count_tokens(  # type: ignore[union-attr]
            {
                "endpoint": self.full_model_name,
                "model": self.full_model_name,
                "contents": contents,
            }
        )
        return response.total_tokens

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> int:
        """Get the number of tokens in the messages.

        Uses the Vertex AI count_tokens API to accurately count tokens,
        including multi-modal content like images and videos.

        Args:
            messages: The list of messages to count tokens for.
            tools: Optional list of tools to include in token count.
                Currently not supported and will be ignored.

        Returns:
            The total number of tokens in the messages.

        Example:
            ```python
            from langchain_core.messages import HumanMessage
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(model="gemini-2.0-flash")

            # Text-only message
            messages = [HumanMessage(content="Hello, world!")]
            token_count = llm.get_num_tokens_from_messages(messages)

            # Multi-modal message with image
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,..."},
                        },
                    ]
                )
            ]
            token_count = llm.get_num_tokens_from_messages(messages)
            ```
        """
        if tools:
            logger.warning(
                "Tool token counting is not yet supported for ChatVertexAI. "
                "The tools parameter will be ignored."
            )
        _, contents = _parse_chat_history_gemini(
            list(messages),
            self._image_bytes_loader_client,
            perform_literal_eval_on_string_raw_content=self.perform_literal_eval_on_string_raw_content,
        )
        response = self.prediction_client.count_tokens(  # type: ignore[union-attr]
            {
                "endpoint": self.full_model_name,
                "model": self.full_model_name,
                "contents": contents,
            }
        )
        return response.total_tokens

    def _tools_gemini(
        self,
        tools: _ToolsType | None = None,
        functions: _ToolsType | None = None,
    ) -> list[GapicTool] | None:
        if tools and functions:
            logger.warning(
                "Binding tools and functions together is not supported.",
                "Only tools will be used",
            )
        if tools:
            return [_format_to_gapic_tool(tools)]
        if functions:
            return [_format_to_gapic_tool(functions)]
        return None

    def _tool_config_gemini(
        self, tool_config: _ToolConfigDict | ToolConfig | None = None
    ) -> GapicToolConfig | None:
        if tool_config and not isinstance(tool_config, ToolConfig):
            return _format_tool_config(tool_config)
        return None

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The `CallbackManager` for LLM run. Not used at the moment.

        Returns:
            The `ChatResult` that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        should_stream = stream is True or (stream is None and self.streaming)

        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        return await self._agenerate_gemini(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from self._stream_gemini(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return

    def _stream_gemini(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)
        timeout = kwargs.pop("timeout", self.timeout)
        response_iter = _completion_with_retry(
            self.prediction_client.stream_generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            timeout=timeout,
            **kwargs,
        )
        total_lc_usage = None
        for response_chunk in response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # TODO: Update to properly support async streaming from gemini.
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)
        timeout = kwargs.pop("timeout", self.timeout)

        response_iter = _acompletion_with_retry(
            self.async_prediction_client.stream_generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            timeout=timeout,
            **kwargs,
        )
        total_lc_usage = None
        async for response_chunk in await response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                await run_manager.on_llm_new_token(chunk.message.content, chunk=chunk)
            yield chunk

    def with_structured_output(
        self,
        schema: dict | type[BaseModel] | type,
        *,
        include_raw: bool = False,
        method: Literal["json_mode"] | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        !!! warning "Behavior changed in `langchain-google-vertexai` 1.1.0"

            Return type corrected in version 1.1.0. Previously if a dict schema was
            provided then the output had the form
            `[{"args": {}, "name": "schema_name"}]` where the output was a list with a
            single dict and the "args" of the one dict corresponded to the schema.

            As of `1.1.0` this has been fixed so that the schema (the value
            corresponding to the old "args" key) is returned directly.

        Args:
            schema: The output schema as a dict or a Pydantic class.

                If a Pydantic class then the model output will be an object of that
                class. If a `dict` then the model output will be a dict. With a Pydantic
                class the returned attributes will be validated, whereas with a `dict`
                they will not be. If `method` is `'function_calling'` and `schema` is a
                `dict`, then the `dict` must match the OpenAI function-calling spec.
            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.
            method: If set to `'json_schema'` it will use controlled generation to
                generate the response rather than function calling.

                Does not work with schemas with references or Pydantic models with
                self-references.

        Returns:
            A `Runnable` that takes any chat model input.

                If `'include_raw'` is `True` then a `dict with` keys:

                * `raw`: `BaseMessage`
                * `parsed`: `_DictOrPydantic | None`
                * `parsing_error`: `BaseException | None`

                If `'include_raw'` is `False`, then just `_DictOrPydantic` is returned,
                where `_DictOrPydantic` depends on the schema.

                If schema is a Pydantic class then `_DictOrPydantic` is the Pydantic
                class.

                If schema is a dict then `_DictOrPydantic` is a dict.

        !!! example "Pydantic schema, exclude raw"
            ```python
            from pydantic import BaseModel
            from langchain_google_vertexai import ChatVertexAI


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
            structured_llm = llm.with_structured_output(AnswerWithJustification)

            structured_llm.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> AnswerWithJustification(
            #     answer='They weigh the same.', justification='A pound is a pound.'
            # )
            ```

        !!! example "Pydantic schema, include raw"
            ```python
            from pydantic import BaseModel
            from langchain_google_vertexai import ChatVertexAI


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
            structured_llm = llm.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_llm.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
            #     'parsing_error': None
            # }
            ```

        !!! example "Dict schema, exclude raw"
            ```python
            from pydantic import BaseModel
            from langchain_core.utils.function_calling import (
                convert_to_openai_function,
            )
            from langchain_google_vertexai import ChatVertexAI


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            dict_schema = convert_to_openai_function(AnswerWithJustification)
            llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
            structured_llm = llm.with_structured_output(dict_schema)

            structured_llm.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            #     'answer': 'They weigh the same',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            # }
            ```

        !!! example "Pydantic schema, streaming"
            ```python
            from pydantic import BaseModel, Field
            from langchain_google_vertexai import ChatVertexAI


            class Explanation(BaseModel):
                '''A topic explanation with examples.'''

                description: str = Field(
                    description="A brief description of the topic."
                )
                examples: str = Field(description="Two examples related to the topic.")


            llm = ChatVertexAI(model_name="gemini-2.0-flash", temperature=0)
            structured_llm = llm.with_structured_output(Explanation, method="json_mode")

            for chunk in structured_llm.stream("Tell me about transformer models"):
                print(chunk)
                print("-------------------------")
            # -> description='Transformer models are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) and are also increasingly used in computer vision and other domains. They rely on the self-attention mechanism to weigh the importance of different parts of the input data, allowing them to effectively capture long-range dependencies. Unlike recurrent neural networks (RNNs), transformers can process the entire input sequence in parallel, leading to significantly faster training times. Key components of transformer models include: the self-attention mechanism (calculates attention weights between different parts of the input), multi-head attention (performs self-attention multiple times with different learned parameters), positional encoding (adds information about the position of tokens in the input sequence), feedforward networks (applies a non-linear transformation to each position), and encoder-decoder structure (used for sequence-to-sequence tasks).' examples='1. BERT (Bidirectional Encoder Representations from Transformers): A pre-trained transformer'
            #    -------------------------
            #    description='Transformer models are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) and are also increasingly used in computer vision and other domains. They rely on the self-attention mechanism to weigh the importance of different parts of the input data, allowing them to effectively capture long-range dependencies. Unlike recurrent neural networks (RNNs), transformers can process the entire input sequence in parallel, leading to significantly faster training times. Key components of transformer models include: the self-attention mechanism (calculates attention weights between different parts of the input), multi-head attention (performs self-attention multiple times with different learned parameters), positional encoding (adds information about the position of tokens in the input sequence), feedforward networks (applies a non-linear transformation to each position), and encoder-decoder structure (used for sequence-to-sequence tasks).' examples='1. BERT (Bidirectional Encoder Representations from Transformers): A pre-trained transformer model that can be fine-tuned for various NLP tasks like text classification, question answering, and named entity recognition. 2. GPT (Generative Pre-trained Transformer): A language model that uses transformers to generate coherent and contextually relevant text. GPT models are used in chatbots, content creation, and code generation.'
            #    -------------------------
            ```
        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        parser: OutputParserLike

        if method == "json_mode":
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                if issubclass(schema, BaseModelV1):
                    schema_json = schema.schema()
                else:
                    schema_json = schema.model_json_schema(mode="serialization")  # type: ignore[attr-defined]
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    schema_json = convert_to_json_schema(schema)
                elif isinstance(schema, dict):
                    schema_json = schema
                else:
                    msg = f"Unsupported schema type {type(schema)}"
                    raise ValueError(msg)
                parser = JsonOutputParser()

            # Resolve refs in schema because they are not supported
            # by the Gemini API.
            schema_json = replace_defs_in_schema(schema_json)

            # API does not support anyOf.
            schema_json = _strip_nullable_anyof(schema_json)

            llm = self.bind(
                response_mime_type="application/json",
                response_schema=schema_json,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema_json,
                },
            )
        else:
            tool_name = _get_tool_name(schema)
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            elif is_typeddict(schema) or isinstance(schema, dict):
                parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
            else:
                msg = f"Unsupported schema type {type(schema)}"
                raise ValueError(msg)
            tool_choice = tool_name

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
        tools: _ToolsType,
        tool_config: _ToolConfigDict | None = None,
        *,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Vertex tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Can be a Pydantic model, `Callable`, or `BaseTool`.

                Pydantic models, `Callable`, and `BaseTool` will be automatically
                converted to their schema dictionary representation.

                Tools with Union types in their arguments are now supported and
                converted to `anyOf` schemas.
            **kwargs: Any additional parameters to pass to the `Runnable` constructor.
        """
        if tool_choice and tool_config:
            msg = (
                "Must specify at most one of tool_choice and tool_config, received "
                f"both:\n\n{tool_choice=}\n\n{tool_config=}"
            )
            raise ValueError(msg)
        try:
            formatted_tools = [convert_to_openai_tool(tool) for tool in tools]  # type: ignore[arg-type]
        except Exception:
            formatted_tools = [_format_to_gapic_tool(tools)]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        elif tool_config:
            kwargs["tool_config"] = tool_config
        else:
            pass
        return self.bind(tools=formatted_tools, **kwargs)

    def _gemini_response_to_chat_result(
        self, response: GenerationResponse
    ) -> ChatResult:
        generations = []
        usage = proto.Message.to_dict(response.usage_metadata)
        lc_usage = _get_usage_metadata_gemini(usage)
        logprobs = self.logprobs if isinstance(self.logprobs, (int, bool)) else False
        for candidate in response.candidates:
            info = get_generation_info(
                candidate, usage_metadata=usage, logprobs=logprobs
            )
            message = _parse_response_candidate(candidate)
            message.response_metadata["model_provider"] = "google_vertexai"
            message.response_metadata["model_name"] = self.model_name
            if "grounding_metadata" in info:
                message.response_metadata["grounding_metadata"] = info.pop(
                    "grounding_metadata"
                )
            if isinstance(message, AIMessage):
                message.usage_metadata = lc_usage
            generations.append(ChatGeneration(message=message, generation_info=info))
        if not response.candidates:
            message = AIMessage(content="")
            message.response_metadata["model_provider"] = "google_vertexai"
            message.response_metadata["model_name"] = self.model_name
            if usage:
                generation_info = {"usage_metadata": usage}
                message.usage_metadata = lc_usage
            else:
                generation_info = {}
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        return ChatResult(generations=generations)

    def _gemini_chunk_to_generation_chunk(
        self,
        response_chunk: GenerationResponse,
        prev_total_usage: UsageMetadata | None = None,
    ) -> tuple[ChatGenerationChunk, UsageMetadata | None]:
        # return an empty completion message if there's no candidates
        usage_metadata = proto.Message.to_dict(response_chunk.usage_metadata)

        # Gather langchain (standard) usage metadata
        # Note: some models (e.g., gemini-1.5-pro with image inputs) return
        # cumulative sums of token counts.
        total_lc_usage = _get_usage_metadata_gemini(usage_metadata)
        if total_lc_usage and prev_total_usage:
            lc_usage: UsageMetadata | None = UsageMetadata(
                input_tokens=total_lc_usage["input_tokens"]
                - prev_total_usage["input_tokens"],
                output_tokens=total_lc_usage["output_tokens"]
                - prev_total_usage["output_tokens"],
                total_tokens=total_lc_usage["total_tokens"]
                - prev_total_usage["total_tokens"],
            )
        else:
            lc_usage = total_lc_usage
        if not response_chunk.candidates:
            message = AIMessageChunk(content="")
            if lc_usage:
                message.usage_metadata = lc_usage
            generation_info = {}
        else:
            top_candidate = response_chunk.candidates[0]
            message = _parse_response_candidate(top_candidate, streaming=True)
            if lc_usage:
                message.usage_metadata = lc_usage
            generation_info = get_generation_info(
                top_candidate,
                usage_metadata={},
            )
            # add model name if final chunk (when we have a definitive finish reason)
            finish_reason = generation_info.get("finish_reason")
            if finish_reason and finish_reason != "FINISH_REASON_UNSPECIFIED":
                message.response_metadata["model_name"] = self.model_name
            # is_blocked is part of "safety_ratings" list
            # but if it's True/False then chunks can't be merged
            generation_info.pop("is_blocked", None)

        message.response_metadata["model_provider"] = "google_vertexai"

        return ChatGenerationChunk(
            message=message,
            generation_info=generation_info,
        ), total_lc_usage


def _get_usage_metadata_gemini(raw_metadata: dict) -> UsageMetadata | None:
    """Get `UsageMetadata` from raw response metadata."""
    input_tokens = raw_metadata.get("prompt_token_count", 0)
    output_tokens = raw_metadata.get("candidates_token_count", 0)
    total_tokens = raw_metadata.get("total_token_count", 0)
    thought_tokens = raw_metadata.get("thoughts_token_count", 0)
    cache_read_tokens = raw_metadata.get("cached_content_token_count", 0)
    if all(
        count == 0
        for count in [input_tokens, output_tokens, total_tokens, cache_read_tokens]
    ):
        return None
    if thought_tokens > 0:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_token_details={"cache_read": cache_read_tokens},
            output_token_details={"reasoning": thought_tokens},
        )
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_token_details={"cache_read": cache_read_tokens},
    )


def _get_tool_name(tool: _ToolType) -> str:
    vertexai_tool = _format_to_gapic_tool([tool])
    return next(f.name for f in vertexai_tool.function_declarations)
