from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import re
import time
import uuid
import warnings
import wave
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from difflib import get_close_matches
from operator import itemgetter
from typing import (
    Any,
    Literal,
    cast,
)

import filetype  # type: ignore[import-untyped]
import proto  # type: ignore[import-untyped]
from google.ai.generativelanguage_v1beta import (
    GenerativeServiceAsyncClient as v1betaGenerativeServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta.types import (
    Blob,
    Candidate,
    CodeExecution,
    CodeExecutionResult,
    Content,
    ExecutableCode,
    FileData,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentRequest,
    GenerateContentResponse,
    GenerationConfig,
    Part,
    SafetySetting,
    ToolConfig,
    VideoMetadata,
)
from google.ai.generativelanguage_v1beta.types import Tool as GoogleTool
from google.api_core.exceptions import (
    FailedPrecondition,
    GoogleAPIError,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
    is_openai_data_block,
)
from langchain_core.language_models.chat_models import BaseChatModel
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
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import Self, is_typeddict

from langchain_google_genai._common import (
    GoogleGenerativeAIError,
    SafetySettingDict,
    _BaseGoogleGenerativeAI,
    get_client_info,
)
from langchain_google_genai._compat import (
    _convert_from_v1_to_generativelanguage_v1beta,
)
from langchain_google_genai._function_utils import (
    _tool_choice_to_tool_config,
    _ToolChoiceType,
    _ToolConfigDict,
    _ToolDict,
    convert_to_genai_function_declarations,
    is_basemodel_subclass_safe,
    tool_to_dict,
)
from langchain_google_genai._image_utils import (
    ImageBytesLoader,
    image_bytes_to_b64_string,
)
from langchain_google_genai.data._profiles import _PROFILES

from . import _genai_extension as genaix

logger = logging.getLogger(__name__)

_allowed_params_prediction_service = ["request", "timeout", "metadata", "labels"]

_FunctionDeclarationType = FunctionDeclaration | dict[str, Any] | Callable[..., Any]

_FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY = (
    "__gemini_function_call_thought_signatures__"
)

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


def _bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _base64_to_bytes(input_str: str) -> bytes:
    return base64.b64decode(input_str.encode("utf-8"))


class ChatGoogleGenerativeAIError(GoogleGenerativeAIError):
    """Custom exception class for errors associated with the `Google GenAI` API.

    This exception is raised when there are specific issues related to the Google GenAI
    API usage in the `ChatGoogleGenerativeAI` class, such as unsupported message types
    or roles.
    """


def _is_gemini_3_or_later(model_name: str) -> bool:
    """Checks if the model is a pre-Gemini 3 model."""
    if not model_name:
        return False
    model_name = model_name.lower()
    if "gemini-3" in model_name:
        return True
    return False


def _is_gemini_25_model(model_name: str) -> bool:
    """Checks if the model is a Gemini 2.5 model."""
    if not model_name:
        return False
    model_name = model_name.lower().replace("models/", "")
    return "gemini-2.5" in model_name


def _create_retry_decorator(
    max_retries: int = 6,
    wait_exponential_multiplier: float = 2.0,
    wait_exponential_min: float = 1.0,
    wait_exponential_max: float = 60.0,
) -> Callable[[Any], Any]:
    """Creates and returns a preconfigured tenacity retry decorator.

    The retry decorator is configured to handle specific Google API exceptions such as
    `ResourceExhausted` and `ServiceUnavailable`. It uses an exponential backoff
    strategy for retries.

    Returns:
        A retry decorator configured for handling specific Google API exceptions.
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=wait_exponential_multiplier,
            min=wait_exponential_min,
            max=wait_exponential_max,
        ),
        retry=(
            retry_if_exception_type(ResourceExhausted)
            | retry_if_exception_type(ServiceUnavailable)
            | retry_if_exception_type(GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _chat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided chat
    generation function. It is useful for handling intermittent issues like network
    errors or temporary service unavailability.

    Args:
        generation_method: The chat generation method to be executed.
        **kwargs: Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator(
        max_retries=kwargs.get("max_retries", 6),
        wait_exponential_multiplier=kwargs.get("wait_exponential_multiplier", 2.0),
        wait_exponential_min=kwargs.get("wait_exponential_min", 1.0),
        wait_exponential_max=kwargs.get("wait_exponential_max", 60.0),
    )

    @retry_decorator
    def _chat_with_retry(**kwargs: Any) -> Any:
        try:
            return generation_method(**kwargs)
        except FailedPrecondition as exc:
            if "location is not supported" in exc.message:
                error_msg = (
                    "Your location is not supported by google-generativeai "
                    "at the moment. Try to use ChatVertexAI LLM from "
                    "langchain_google_vertexai."
                )
                raise ValueError(error_msg)

        except InvalidArgument as e:
            msg = f"Invalid argument provided to Gemini: {e}"
            raise ChatGoogleGenerativeAIError(msg) from e
        except ResourceExhausted as e:
            # Handle quota-exceeded error with recommended retry delay
            if hasattr(e, "retry_after") and getattr(e, "retry_after", 0) < kwargs.get(
                "wait_exponential_max", 60.0
            ):
                time.sleep(e.retry_after)
            raise
        except Exception:
            raise

    params = {
        k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service
    }
    return _chat_with_retry(**params)


async def _achat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided chat
    generation function. It is useful for handling intermittent issues like network
    errors or temporary service unavailability.

    Args:
        generation_method: The chat generation method to be executed.
        **kwargs: Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator(
        max_retries=kwargs.get("max_retries", 6),
        wait_exponential_multiplier=kwargs.get("wait_exponential_multiplier", 2.0),
        wait_exponential_min=kwargs.get("wait_exponential_min", 1.0),
        wait_exponential_max=kwargs.get("wait_exponential_max", 60.0),
    )

    @retry_decorator
    async def _achat_with_retry(**kwargs: Any) -> Any:
        try:
            return await generation_method(**kwargs)
        except InvalidArgument as e:
            # Do not retry for these errors.
            msg = f"Invalid argument provided to Gemini: {e}"
            raise ChatGoogleGenerativeAIError(msg) from e
        except ResourceExhausted as e:
            # Handle quota-exceeded error with recommended retry delay
            if hasattr(e, "retry_after") and getattr(e, "retry_after", 0) < kwargs.get(
                "wait_exponential_max", 60.0
            ):
                time.sleep(e.retry_after)
            raise
        except Exception:
            raise

    params = {
        k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service
    }
    return await _achat_with_retry(**params)


def _convert_to_parts(
    raw_content: str | Sequence[str | dict],
    model: str | None = None,
) -> list[Part]:
    """Converts LangChain message content into `generativelanguage_v1beta` parts.

    Used when preparing Human, System and AI messages for sending to the API.

    Handles both legacy (pre-v1) dict-based content blocks and v1 `ContentBlock`
    objects.
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
                    # Check if there's a signature attached to this text block
                    thought_sig = None
                    if "extras" in part and isinstance(part["extras"], dict):
                        sig = part["extras"].get("signature")
                        if sig and isinstance(sig, str):
                            # Decode base64-encoded signature back to bytes
                            thought_sig = base64.b64decode(sig)
                    if thought_sig:
                        parts.append(
                            Part(text=part["text"], thought_signature=thought_sig)
                        )
                    else:
                        parts.append(Part(text=part["text"]))
                elif is_data_content_block(part):
                    # Handle both legacy LC blocks (with `source_type`) and blocks >= v1

                    if "source_type" in part:
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
                    inline_data: dict = {"data": bytes_}
                    if "mime_type" in part:
                        inline_data["mime_type"] = part["mime_type"]
                    else:
                        # Guess MIME type based on data field if not provided
                        source = cast(
                            "str",
                            part.get("url") or part.get("base64") or part.get("data"),
                        )
                        mime_type, _ = mimetypes.guess_type(source)
                        if not mime_type:
                            # Last resort - try to guess based on file bytes
                            kind = filetype.guess(bytes_)
                            if kind:
                                mime_type = kind.mime
                        if mime_type:
                            inline_data["mime_type"] = mime_type

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
                            inline_data["media_resolution"] = part["media_resolution"]

                    parts.append(Part(inline_data=inline_data))
                elif part["type"] == "image_url":
                    # Chat Completions image format
                    img_url = part["image_url"]
                    if isinstance(img_url, dict):
                        if "url" not in img_url:
                            msg = f"Unrecognized message image format: {img_url}"
                            raise ValueError(msg)
                        img_url = img_url["url"]
                    parts.append(image_loader.load_part(img_url))
                elif part["type"] == "media":
                    # Handle `media` following pattern established in LangChain.js
                    # https://github.com/langchain-ai/langchainjs/blob/e536593e2585f1dd7b0afc187de4d07cb40689ba/libs/langchain-google-common/src/utils/gemini.ts#L93-L106
                    if "mime_type" not in part:
                        msg = f"Missing mime_type in media part: {part}"
                        raise ValueError(msg)
                    mime_type = part["mime_type"]
                    media_part = Part()

                    if "data" in part:
                        # Embedded media
                        media_part.inline_data = Blob(
                            data=part["data"], mime_type=mime_type
                        )
                    elif "file_uri" in part:
                        # Referenced files (e.g. stored in GCS)
                        media_part.file_data = FileData(
                            file_uri=part["file_uri"], mime_type=mime_type
                        )
                    else:
                        msg = f"Media part must have either data or file_uri: {part}"
                        raise ValueError(msg)
                    if "video_metadata" in part:
                        metadata = VideoMetadata(part["video_metadata"])
                        media_part.video_metadata = metadata

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
                            if media_part.inline_data:
                                media_part.inline_data.media_resolution = part[
                                    "media_resolution"
                                ]
                            elif media_part.file_data:
                                media_part.file_data.media_resolution = part[
                                    "media_resolution"
                                ]

                    parts.append(media_part)
                elif part["type"] == "thinking":
                    # Pre-existing thinking block format that we continue to store as
                    thought_sig = None
                    if "signature" in part:
                        sig = part["signature"]
                        if sig and isinstance(sig, str):
                            # Decode base64-encoded signature back to bytes
                            thought_sig = base64.b64decode(sig)
                    parts.append(
                        Part(
                            text=part["thinking"],
                            thought=True,
                            thought_signature=thought_sig,
                        )
                    )
                elif part["type"] == "reasoning":
                    # ReasoningContentBlock (when output_version = "v1")
                    extras = part.get("extras", {}) or {}
                    sig = extras.get("signature")
                    thought_sig = None
                    if sig and isinstance(sig, str):
                        # Decode base64-encoded signature back to bytes
                        thought_sig = base64.b64decode(sig)
                    parts.append(
                        Part(
                            text=part["reasoning"],
                            thought=True,
                            thought_signature=thought_sig,
                        )
                    )
                elif part["type"] == "server_tool_call":
                    if part.get("name") == "code_interpreter":
                        args = part.get("args", {})
                        code = args.get("code", "")
                        language = args.get("language", "python")
                        executable_code_part = Part(
                            executable_code=ExecutableCode(language=language, code=code)
                        )
                        parts.append(executable_code_part)
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
                    output = part.get("output", "")
                    status = part.get("status", "success")
                    # Map status to outcome: success → 1 (OUTCOME_OK), error → 2
                    outcome = 1 if status == "success" else 2
                    # Check extras for original outcome if available
                    if "extras" in part and "outcome" in part["extras"]:
                        outcome = part["extras"]["outcome"]
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
                        outcome = part["outcome"]
                    else:
                        # Backward compatibility
                        outcome = 1  # Default to success if not specified
                    code_execution_result_part = Part(
                        code_execution_result=CodeExecutionResult(
                            output=part["code_execution_result"], outcome=outcome
                        )
                    )
                    parts.append(code_execution_result_part)
                else:
                    msg = f"Unrecognized message part type: {part['type']}."
                    raise ValueError(msg)
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


# To generate the below thought signature:

# from langchain_google_genai import ChatGoogleGenerativeAI
#
# def generate_placeholder_thoughts(value: int) -> str:
#     """Placeholder tool."""
#     pass
#
# model = ChatGoogleGenerativeAI(
#     model="gemini-3-pro-preview"
# ).bind_tools([generate_placeholder_thoughts])
#
# response = model.invoke("Generate a placeholder tool invocation.")

DUMMY_THOUGHT_SIGNATURE = _base64_to_bytes(
    "ErQCCrECAdHtim8MtxgeMCRCiNiyoyImxtYAEDzz4NXOr/HSL3rA7rPPvHWZCm+T9VSDYh/mt9lESoH4wQh"
    "/ca1zDtWTN6XOL1+S3krYLQeqp47RV/b1eSq5jdZF28S4Lb7w4A3/EFdybc4SFb2/YhMm+CulYLmLA4Tr4V"
    "Su0eMWgxM3HVt6u0jECf5BbXzj0qjJ32tEQYJvKvV8H1tCHvB6J+RZhsDr+TcyOCaqxDoR4WKxXYxNRZb3h"
    "YTuCnBEDPhn1lROumVaghi9nEIgc17z002zLoyqIptlLfIVw70FXkCLsPUSL1SjPQYtGL8PVncVajeqGogR"
    "D/eZSVZ1Zr5tshxh3DQ+JAYNcrHaRHWC4Hg0H6oftYx+JdJD9B/81NYV9jyGxP7zHKFHOELl0IUP5GEXP9I"
    "="
)


def _parse_chat_history(
    input_messages: Sequence[BaseMessage],
    convert_system_message_to_human: bool = False,
    model: str | None = None,
) -> tuple[Content | None, list[Content]]:
    """Parses sequence of `BaseMessage` into system instruction and formatted messages.

    Args:
        input_messages: Sequence of `BaseMessage` objects representing the chat history.
        convert_system_message_to_human: Whether to convert the first system message
            into a `HumanMessage`. Deprecated, use system instructions instead.
        model: The model name, used for version-specific logic.

    Returns:
        A tuple containing:

            - An optional `google.ai.generativelanguage_v1beta.types.Content`
                representing the system instruction (if any).
            - A list of `google.ai.generativelanguage_v1beta.types.Content` representing
                the formatted messages.
    """
    if convert_system_message_to_human:
        warnings.warn(
            "The 'convert_system_message_to_human' parameter is deprecated and will be "
            "removed in a future version. Use system instructions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    input_messages = list(input_messages)  # Make a mutable copy

    # Case where content was serialized to v1 format
    for idx, message in enumerate(input_messages):
        if (
            isinstance(message, AIMessage)
            and message.response_metadata.get("output_version") == "v1"
        ):
            # Unpack known v1 content to v1beta format for the request
            #
            # Old content types and any previously serialized messages passed back in to
            # history will skip this, but hit and processed in `_convert_to_parts`
            input_messages[idx] = message.model_copy(
                update={
                    "content": _convert_from_v1_to_generativelanguage_v1beta(
                        cast("list[types.ContentBlock]", message.content),
                        message.response_metadata.get("model_provider"),
                    )
                }
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
                system_instruction.parts.extend(system_parts)
            else:
                pass
            continue
        if isinstance(message, AIMessage):
            role = "model"
            if message.tool_calls:
                ai_message_parts = []
                function_call_sigs: dict[Any, str] = message.additional_kwargs.get(
                    _FUNCTION_CALL_THOUGHT_SIGNATURES_MAP_KEY, {}
                )
                for tool_call_idx, tool_call in enumerate(message.tool_calls):
                    function_call = FunctionCall(
                        {
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                        }
                    )
                    # Check if there's a signature for this function call
                    sig = function_call_sigs.get(tool_call.get("id"))
                    if sig:
                        ai_message_parts.append(
                            Part(
                                function_call=function_call,
                                thought_signature=_base64_to_bytes(sig),
                            )
                        )
                    else:
                        ai_message_parts.append(Part(function_call=function_call))
                tool_messages_parts = _get_ai_message_tool_messages_parts(
                    tool_messages=tool_messages, ai_message=message, model=model
                )
                formatted_messages.append(Content(role=role, parts=ai_message_parts))
                formatted_messages.append(
                    Content(role="user", parts=tool_messages_parts)
                )
                continue
            if raw_function_call := message.additional_kwargs.get("function_call"):
                function_call = FunctionCall(
                    {
                        "name": raw_function_call["name"],
                        "args": json.loads(raw_function_call["arguments"]),
                    }
                )
                parts = [Part(function_call=function_call)]
            elif message.response_metadata.get("output_version") == "v1":
                # Already converted to v1beta format above
                parts = message.content  # type: ignore[assignment]
            else:
                # Prepare request content parts from message.content field
                parts = _convert_to_parts(message.content, model=model)
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message.content, model=model)
            if i == 1 and convert_system_message_to_human and system_instruction:
                parts = list(system_instruction.parts) + parts
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
            msg = formatted_messages[i]
            if msg.role == "user":
                has_function_response = False
                has_standard_content = False
                for part in msg.parts:
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
        # but lacks a thought signature, inject a dummy value. This satisfies the
        # API's schema validation without requiring the original internal thought data.
        start_idx = active_loop_start_idx + 1 if active_loop_start_idx != -1 else 0
        for i in range(start_idx, len(formatted_messages)):
            msg = formatted_messages[i]
            if msg.role == "model":
                first_fc_seen = False
                for part in msg.parts:
                    if part.function_call:
                        if not first_fc_seen:
                            if not part.thought_signature:
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


def _parse_response_candidate(
    response_candidate: Candidate,
    streaming: bool = False,
    model_name: str | None = None,
) -> AIMessage:
    content: None | str | list[str | dict] = None
    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {"model_provider": "google_genai"}
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []
    for part in response_candidate.content.parts:
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
            or ("text" in part and thought_sig)  # text part with thought signature
        ):
            text_block: dict[str, Any] = {"type": "text", "text": text or ""}
            if thought_sig:
                text_block["extras"] = {"signature": thought_sig}
            if thought_sig or _is_gemini_3_or_later(model_name or ""):
                # append blocks if there's a signature or new Gemini model
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
            # outcome: 1 = OUTCOME_OK (success), else = error
            outcome = part.code_execution_result.outcome
            execution_result = {
                "type": "code_execution_result",
                "code_execution_result": part.code_execution_result.output,
                "outcome": outcome,
                "tool_call_id": "",  # Linked via block translator
            }
            content = _append_to_content(content, execution_result)

        if (
            hasattr(part, "inline_data")
            and part.inline_data
            and part.inline_data.mime_type.startswith("audio/")
        ):
            buffer = io.BytesIO()

            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                # TODO: Read Sample Rate from MIME content type.
                wf.setframerate(24000)
                wf.writeframes(part.inline_data.data)

            audio_data = buffer.getvalue()
            additional_kwargs["audio"] = audio_data

            # For backwards compatibility, audio stays in additional_kwargs by default
            # and is accessible via .content_blocks property

        if (
            hasattr(part, "inline_data")
            and part.inline_data
            and part.inline_data.mime_type.startswith("image/")
        ):
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

        if part.function_call:
            function_call = {"name": part.function_call.name}
            # dump to match other function calling llm for now
            function_call_args_dict = proto.Message.to_dict(part.function_call)["args"]

            # Fix: Correct integer-like floats from protobuf conversion
            # The protobuf library sometimes converts integers to floats
            corrected_args = {
                k: int(v) if isinstance(v, float) and v.is_integer() else v
                for k, v in function_call_args_dict.items()
            }

            function_call["arguments"] = json.dumps(corrected_args)
            additional_kwargs["function_call"] = function_call

            tool_call_id = function_call.get("id", str(uuid.uuid4()))
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
        if _is_gemini_3_or_later(model_name or ""):
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


def _response_to_result(
    response: GenerateContentResponse,
    stream: bool = False,
    prev_usage: UsageMetadata | None = None,
) -> ChatResult:
    """Converts a PaLM API response into a LangChain `ChatResult`."""
    llm_output = {"prompt_feedback": proto.Message.to_dict(response.prompt_feedback)}

    # Get usage metadata
    try:
        input_tokens = response.usage_metadata.prompt_token_count
        thought_tokens = response.usage_metadata.thoughts_token_count
        output_tokens = response.usage_metadata.candidates_token_count + thought_tokens
        total_tokens = response.usage_metadata.total_token_count
        cache_read_tokens = response.usage_metadata.cached_content_token_count
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

    for candidate in response.candidates:
        generation_info = {}
        if candidate.finish_reason:
            generation_info["finish_reason"] = candidate.finish_reason.name
            # Add model_name in last chunk
            generation_info["model_name"] = response.model_version
        generation_info["safety_ratings"] = [
            proto.Message.to_dict(safety_rating, use_integers_for_enums=False)
            for safety_rating in candidate.safety_ratings
        ]
        message = _parse_response_candidate(
            candidate, streaming=stream, model_name=response.model_version
        )

        if not hasattr(message, "response_metadata"):
            message.response_metadata = {}

        try:
            if candidate.grounding_metadata:
                grounding_metadata = proto.Message.to_dict(candidate.grounding_metadata)
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
        if stream:
            generations = [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={
                            "prompt_feedback": proto.Message.to_dict(
                                response.prompt_feedback
                            )
                        },
                    ),
                    generation_info={},
                )
            ]
        else:
            generations = [ChatGeneration(message=AIMessage(""), generation_info={})]
    return ChatResult(generations=generations, llm_output=llm_output)


def _is_event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class ChatGoogleGenerativeAI(_BaseGoogleGenerativeAI, BaseChatModel):
    r"""Google GenAI chat model integration.

    Instantiation:
        To use, you must have either:

        1. The `GOOGLE_API_KEY` environment variable set with your API key, or
        2. Pass your API key using the `google_api_key` kwarg to the
            `ChatGoogleGenerativeAI` constructor.

        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        llm.invoke("Write me a ballad about LangChain")
        ```

    Invoke:
        ```python
        messages = [
            ("system", "Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        llm.invoke(messages)
        ```

        ```python
        AIMessage(
            content="J'adore programmer. \\n",
            response_metadata={
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
            },
            id="run-56cecc34-2e54-4b52-a974-337e47008ad2-0",
            usage_metadata={
                "input_tokens": 18,
                "output_tokens": 5,
                "total_tokens": 23,
            },
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

        ```python
        stream = llm.stream(messages)
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

    Async:
        ```python
        await llm.ainvoke(messages)

        # stream:
        async for chunk in (await llm.astream(messages))

        # batch:
        await llm.abatch([messages])
        ```

    Context caching:
        Context caching allows you to store and reuse content (e.g., PDFs, images) for
        faster processing. The `cached_content` parameter accepts a cache name created
        via the Google Generative AI API.

        Below are two examples: caching a single file directly and caching multiple
        files using `Part`.

        !!! example "Single file example"

            This caches a single file and queries it.

            ```python
            from google import genai
            from google.genai import types
            import time
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            client = genai.Client()

            # Upload file
            file = client.files.upload(file="./example_file")
            while file.state.name == "PROCESSING":
                time.sleep(2)
                file = client.files.get(name=file.name)

            # Create cache
            model = "models/gemini-2.5-flash"
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

        !!! example "Multiple files example"

            This caches two files using `Part` and queries them together.

            ```python
            from google import genai
            from google.genai.types import CreateCachedContentConfig, Content, Part
            import time
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

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
            model = "gemini-2.5-flash"
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

    Search:
        ```python
        from google.ai.generativelanguage_v1beta.types import Tool as GenAITool

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        resp = llm.invoke(
            "When is the next total solar eclipse in US?",
            tools=[GenAITool(google_search={})],
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
                description="How funny the joke is, from 1 to 10"
            )


        # Default method uses function calling
        structured_llm = llm.with_structured_output(Joke)

        # For more reliable output, use json_schema with native responseSchema
        structured_llm_json = llm.with_structured_output(Joke, method="json_schema")
        structured_llm_json.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup="Why are cats so good at video games?",
            punchline="They have nine lives on the internet",
            rating=None,
        )
        ```

        Two methods are supported for structured output:

        * `method='function_calling'` (default): Uses tool calling to extract
            structured data. Compatible with all models.
        * `method='json_schema'`: Uses Gemini's native structured output.

            Supports unions (`anyOf`), recursive schemas (`$ref`), property ordering
            preservation, and streaming of partial JSON chunks.

            Uses Gemini's `response_json_schema` API param. Refer to the Gemini API
            [docs](https://ai.google.dev/gemini-api/docs/structured-output) for more
            details.

        The `json_schema` method is recommended for better reliability as it
        constrains the model's generation process directly rather than relying on
        post-processing tool calls.

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
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        The weather in this image appears to be sunny and pleasant. The sky is a bright
        blue with scattered white clouds, suggesting fair weather. The lush green grass
        and trees indicate a warm and possibly slightly breezy day. There are no...
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
                    "source_type": "base64",
                    "mime_type": "application/pdf",
                    "data": pdf_base64,
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
                    "source_type": "base64",
                    "mime_type": "video/mp4",
                    "data": video_base64,
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
        text that is highlighted on the screen, defines the word...
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
                    "source_type": "base64",
                    "mime_type": "audio/mp3",
                    "data": audio_base64,
                },
            ]
        )
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        In this episode of the Made by Google podcast, Stephen Johnson and Simon
        Tokumine discuss NotebookLM, a tool designed to help users understand complex
        material in various modalities, with a focus on its unexpected uses, the...
        ```

    File upload:
        You can also upload files to Google's servers and reference them by URI.

        This works for PDFs, images, videos, and audio files.

        ```python
        import time
        from google import genai
        from langchain_core.messages import HumanMessage

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
        ai_msg = llm.invoke([message])
        ai_msg.content
        ```

        ```txt
        This research paper assesses and mitigates multi-turn jailbreak vulnerabilities
        in large language models using the Crescendo attack study, evaluating attack
        success rates and mitigation strategies like prompt...
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
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            include_thoughts=True,
        )
        ai_msg = llm.invoke("How many 'r's are in the word 'strawberry'?")
        ```

    Token usage:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {"input_tokens": 18, "output_tokens": 5, "total_tokens": 23}
        ```

    Response metadata:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.response_metadata
        ```

        ```python
        {
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
    """

    thinking_level: Literal["low", "high"] | None = Field(
        default=None,
    )
    """Indicates the thinking level.

    Supported values:
        * `'low'`: Minimizes latency and cost.
        * `'high'`: Maximizes reasoning depth.

    !!! note "Replaces `thinking_budget`"

        `thinking_budget` is deprecated for Gemini 3+ models. If both parameters are
        provided, `thinking_level` takes precedence.

        If left unspecified, the model's default thinking level is used. For Gemini 3+,
        this defaults to `'high'`.
    """

    client: Any = Field(default=None, exclude=True)

    async_client_running: Any = Field(default=None, exclude=True)

    default_metadata: Sequence[tuple[str, str]] | None = Field(
        default=None, alias="default_metadata_input"
    )

    convert_system_message_to_human: bool = False
    """Whether to merge any leading `SystemMessage` into the following `HumanMessage`.

    Gemini does not support system messages; any unsupported messages will raise an
    error.
    """

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

    The format of the dictionary should follow Open API schema.

    Has JSON Schema support including:

    - `anyOf` for unions
    - `$ref` for recursive schemas
    - Output property ordering
    - Minimum/maximum constraints
    - Streaming of partial JSON chunks

    Refer to the Gemini API [docs](https://ai.google.dev/gemini-api/docs/structured-output)
    for more details.
    """

    cached_content: str | None = None
    """The name of the cached content used as context to serve the prediction.

    !!! note

        Only used in explicit caching, where users can have control over caching (e.g.
        what content to cache) and enjoy guaranteed cost savings. Format:
        `cachedContents/{cachedContent}`.
    """

    stop: list[str] | None = None
    """Stop sequences for the model."""

    streaming: bool | None = None
    """Whether to stream responses from the model."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any unexpected initialization parameters."""

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
    def lc_secrets(self) -> dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai"

    @property
    def _supports_code_execution(self) -> bool:
        """Whether the model supports code execution.

        See the [Gemini models docs](https://ai.google.dev/gemini-api/docs/models) for a
        full list.
        """
        return (
            "gemini-1.5-pro" in self.model
            or "gemini-1.5-flash" in self.model
            or "gemini-2" in self.model
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and passes them to `google-generativeai` package."""
        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            msg = "temperature must be in the range [0.0, 2.0]"
            raise ValueError(msg)

        if "temperature" not in self.model_fields_set and _is_gemini_3_or_later(
            self.model
        ):
            self.temperature = 1.0

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            msg = "top_p must be in the range [0.0, 1.0]"
            raise ValueError(msg)

        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        if not any(self.model.startswith(prefix) for prefix in ("models/",)):
            self.model = f"models/{self.model}"

        additional_headers = self.additional_headers or {}
        self.default_metadata = tuple(additional_headers.items())
        client_info = get_client_info(f"ChatGoogleGenerativeAI:{self.model}")
        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key
        transport: str | None = self.transport

        # Merge base_url into client_options if provided
        client_options = self.client_options or {}
        if self.base_url and "api_endpoint" not in client_options:
            client_options = {**client_options, "api_endpoint": self.base_url}

        self.client = genaix.build_generative_service(
            credentials=self.credentials,
            api_key=google_api_key,
            client_info=client_info,
            client_options=client_options,
            transport=transport,
        )
        self.async_client_running = None
        return self

    @property
    def async_client(self) -> v1betaGenerativeServiceAsyncClient:
        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key
        # NOTE: genaix.build_generative_async_service requires
        # a running event loop, which causes an error
        # when initialized inside a ThreadPoolExecutor.
        # this check ensures that async client is only initialized
        # within an asyncio event loop to avoid the error
        if not self.async_client_running and _is_event_loop_running():
            # async clients don't support "rest" transport
            # https://github.com/googleapis/gapic-generator-python/issues/1962

            # However, when using custom endpoints, we can try to keep REST transport
            transport = self.transport
            client_options = self.client_options or {}

            # Check for custom endpoint
            has_custom_endpoint = self.base_url or (
                self.client_options
                and "api_endpoint" in self.client_options
                and self.client_options["api_endpoint"]
                != "https://generativelanguage.googleapis.com"
            )

            # Only change to grpc_asyncio if no custom endpoint is specified
            if transport == "rest" and not has_custom_endpoint:
                transport = "grpc_asyncio"

            # Merge base_url into client_options if provided
            if self.base_url and "api_endpoint" not in client_options:
                client_options = {**client_options, "api_endpoint": self.base_url}

            self.async_client_running = genaix.build_generative_async_service(
                credentials=self.credentials,
                api_key=google_api_key,
                client_info=get_client_info(f"ChatGoogleGenerativeAI:{self.model}"),
                client_options=client_options,
                transport=transport,
            )
        return self.async_client_running

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden."""
        if self.profile is None:
            model_id = re.sub(r"-\d{3}$", "", self.model.replace("models/", ""))
            self.profile = _get_default_model_profile(model_id)
        return self

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "n": self.n,
            "safety_settings": self.safety_settings,
            "response_modalities": self.response_modalities,
            "media_resolution": self.media_resolution,
            "thinking_budget": self.thinking_budget,
            "include_thoughts": self.include_thoughts,
            "thinking_level": self.thinking_level,
        }

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        code_execution: bool | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Override `invoke` on `ChatGoogleGenerativeAI` to add `code_execution`.

        See the [models page](https://ai.google.dev/gemini-api/docs/models) to see if
        your chosen model supports code execution. When enabled, the model can execute
        code to solve problems.
        """
        if code_execution is not None:
            if not self._supports_code_execution:
                msg = (
                    "Code execution is only supported on Gemini 1.5, 2.0, and 2.5 "
                    f"models. Current model: {self.model}"
                )
                raise ValueError(msg)
            if "tools" not in kwargs:
                code_execution_tool = GoogleTool(code_execution=CodeExecution())
                kwargs["tools"] = [code_execution_tool]

            else:
                msg = "Tools are already defined.code_execution tool can't be defined"
                raise ValueError(msg)

        return super().invoke(input, config, stop=stop, **kwargs)

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        models_prefix = "models/"
        ls_model_name = (
            self.model[len(models_prefix) :]
            if self.model and self.model.startswith(models_prefix)
            else self.model
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

    def _prepare_params(
        self,
        stop: list[str] | None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationConfig:
        # Extract thinking parameters with kwargs override
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        thinking_level = kwargs.get("thinking_level", self.thinking_level)

        if thinking_level is not None and thinking_budget is not None:
            msg = (
                "Both 'thinking_level' and 'thinking_budget' were specified. "
                "'thinking_level' is not yet supported by the current API version, "
                "so 'thinking_budget' will be used instead."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        gen_config = {
            k: v
            for k, v in {
                "candidate_count": self.n,
                "temperature": self.temperature,
                "stop_sequences": stop,
                "max_output_tokens": kwargs.get(
                    "max_output_tokens", self.max_output_tokens
                ),
                "top_k": self.top_k,
                "top_p": self.top_p,
                "response_modalities": self.response_modalities,
                "thinking_config": (
                    (
                        (
                            {"thinking_budget": thinking_budget}
                            if thinking_budget is not None
                            else {}
                        )
                        | (
                            {"include_thoughts": self.include_thoughts}
                            if self.include_thoughts is not None
                            else {}
                        )
                        | (
                            {"thinking_level": thinking_level}
                            if thinking_level is not None
                            else {}
                        )
                    )
                    if thinking_budget is not None
                    or self.include_thoughts is not None
                    or thinking_level is not None
                    else None
                ),
            }.items()
            if v is not None
        }
        if generation_config:
            gen_config = {**gen_config, **generation_config}

        response_mime_type = kwargs.get("response_mime_type", self.response_mime_type)
        if response_mime_type is not None:
            gen_config["response_mime_type"] = response_mime_type

        response_schema = kwargs.get("response_schema", self.response_schema)

        # In case passed in as a direct kwarg
        response_json_schema = kwargs.get("response_json_schema")

        # Handle both response_schema and response_json_schema
        # (Regardless, we use `response_json_schema` in the request)
        schema_to_use = (
            response_json_schema
            if response_json_schema is not None
            else response_schema
        )

        if schema_to_use is not None:
            if response_mime_type != "application/json":
                param_name = (
                    "response_json_schema"
                    if response_json_schema is not None
                    else "response_schema"
                )
                error_message = (
                    f"'{param_name}' is only supported when "
                    f"response_mime_type is set to 'application/json'"
                )
                if response_mime_type == "text/x.enum":
                    error_message += (
                        ". Instead of 'text/x.enum', define enums using JSON schema."
                    )
                raise ValueError(error_message)

            gen_config["response_json_schema"] = schema_to_use

        media_resolution = kwargs.get("media_resolution", self.media_resolution)
        if media_resolution is not None:
            gen_config["media_resolution"] = media_resolution

        return GenerationConfig(**gen_config)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | _ToolConfigDict | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
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
        if self.timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = self.max_retries
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            **kwargs,
            generation_method=self.client.generate_content,
            metadata=self.default_metadata,
        )
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
        tool_config: dict | _ToolConfigDict | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.async_client:
            updated_kwargs = {
                **kwargs,
                "tools": tools,
                "functions": functions,
                "safety_settings": safety_settings,
                "tool_config": tool_config,
                "generation_config": generation_config,
            }
            return await super()._agenerate(
                messages, stop, run_manager, **updated_kwargs
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
        if self.timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = self.max_retries
        response: GenerateContentResponse = await _achat_with_retry(
            request=request,
            **kwargs,
            generation_method=self.async_client.generate_content,
            metadata=self.default_metadata,
        )
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
        tool_config: dict | _ToolConfigDict | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
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
        if self.timeout is not None and "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = self.max_retries
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            generation_method=self.client.stream_generate_content,
            **kwargs,
            metadata=self.default_metadata,
        )

        prev_usage_metadata: UsageMetadata | None = None  # cumulative usage
        index = -1
        index_type = ""
        for chunk in response:
            _chat_result = _response_to_result(
                chunk, stream=True, prev_usage=prev_usage_metadata
            )
            gen = cast("ChatGenerationChunk", _chat_result.generations[0])
            message = cast("AIMessageChunk", gen.message)

            # populate index if missing
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, dict) and "type" in block:
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index = index + 1
                        if "index" not in block:
                            block["index"] = index

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
        tool_config: dict | _ToolConfigDict | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if not self.async_client:
            updated_kwargs = {
                **kwargs,
                "tools": tools,
                "functions": functions,
                "safety_settings": safety_settings,
                "tool_config": tool_config,
                "generation_config": generation_config,
            }
            async for value in super()._astream(
                messages, stop, run_manager, **updated_kwargs
            ):
                yield value
        else:
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
            if self.timeout is not None and "timeout" not in kwargs:
                kwargs["timeout"] = self.timeout
            if "max_retries" not in kwargs:
                kwargs["max_retries"] = self.max_retries
            prev_usage_metadata: UsageMetadata | None = None  # cumulative usage

            index = -1
            index_type = ""
            async for chunk in await _achat_with_retry(
                request=request,
                generation_method=self.async_client.stream_generate_content,
                **kwargs,
                metadata=self.default_metadata,
            ):
                _chat_result = _response_to_result(
                    chunk, stream=True, prev_usage=prev_usage_metadata
                )
                gen = cast("ChatGenerationChunk", _chat_result.generations[0])
                message = cast("AIMessageChunk", gen.message)

                # populate index if missing
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, dict) and "type" in block:
                            if block["type"] != index_type:
                                index_type = block["type"]
                                index = index + 1
                            if "index" not in block:
                                block["index"] = index

                prev_usage_metadata = (
                    message.usage_metadata
                    if prev_usage_metadata is None
                    else add_usage(prev_usage_metadata, message.usage_metadata)
                )

                if run_manager:
                    await run_manager.on_llm_new_token(gen.text, chunk=gen)
                yield gen

    def _prepare_request(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        tools: Sequence[_ToolDict | GoogleTool] | None = None,
        functions: Sequence[_FunctionDeclarationType] | None = None,
        safety_settings: SafetySettingDict | None = None,
        tool_config: dict | _ToolConfigDict | None = None,
        tool_choice: _ToolChoiceType | bool | None = None,
        generation_config: dict[str, Any] | None = None,
        cached_content: str | None = None,
        **kwargs: Any,
    ) -> GenerateContentRequest:
        if tool_choice and tool_config:
            msg = (
                "Must specify at most one of tool_choice and tool_config, received "
                f"both:\n\n{tool_choice=}\n\n{tool_config=}"
            )
            raise ValueError(msg)

        formatted_tools = None
        code_execution_tool = GoogleTool(code_execution=CodeExecution())
        if tools == [code_execution_tool]:
            formatted_tools = tools
        elif tools:
            formatted_tools = [convert_to_genai_function_declarations(tools)]
        elif functions:
            formatted_tools = [convert_to_genai_function_declarations(functions)]

        filtered_messages = []
        for message in messages:
            if isinstance(message, HumanMessage) and not message.content:
                warnings.warn(
                    "HumanMessage with empty content was removed to prevent API error"
                )
            else:
                filtered_messages.append(message)
        messages = filtered_messages

        if self.convert_system_message_to_human:
            system_instruction, history = _parse_chat_history(
                messages,
                convert_system_message_to_human=self.convert_system_message_to_human,
                model=self.model,
            )
        else:
            system_instruction, history = _parse_chat_history(
                messages,
                model=self.model,
            )

        # Validate that we have at least one content message for the API
        if not history:
            msg = (
                "No content messages found. The Gemini API requires at least one "
                "non-system message (HumanMessage, AIMessage, etc.) in addition to "
                "any SystemMessage. Please include additional messages in your input."
            )
            raise ValueError(msg)

        if tool_choice:
            if not formatted_tools:
                msg = (
                    f"Received {tool_choice=} but no {tools=}. 'tool_choice' can only "
                    f"be specified if 'tools' is specified."
                )
                raise ValueError(msg)
            all_names: list[str] = []
            for t in formatted_tools:
                if hasattr(t, "function_declarations"):
                    t_with_declarations = cast("Any", t)
                    all_names.extend(
                        f.name for f in t_with_declarations.function_declarations
                    )
                elif isinstance(t, GoogleTool) and hasattr(t, "code_execution"):
                    continue
                else:
                    msg = f"Tool {t} doesn't have function_declarations attribute"
                    raise TypeError(msg)

            tool_config = _tool_choice_to_tool_config(tool_choice, all_names)

        formatted_tool_config = None
        if tool_config:
            formatted_tool_config = ToolConfig(
                function_calling_config=tool_config["function_calling_config"]
            )
        formatted_safety_settings = []
        if safety_settings:
            formatted_safety_settings = [
                SafetySetting(category=c, threshold=t)
                for c, t in safety_settings.items()
            ]
        request = GenerateContentRequest(
            model=self.model,
            contents=history,  # google.ai.generativelanguage_v1beta.types.Content
            tools=formatted_tools,
            tool_config=formatted_tool_config,
            safety_settings=formatted_safety_settings,
            generation_config=self._prepare_params(
                stop,
                generation_config=generation_config,
                **kwargs,
            ),
            cached_content=cached_content,
        )
        if system_instruction:
            request.system_instruction = system_instruction

        return request

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text. Uses the model's tokenizer.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.

        Example:
            ```python
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            num_tokens = llm.get_num_tokens("Hello, world!")
            print(num_tokens)
            # 4
            ```
        """
        result = self.client.count_tokens(
            model=self.model, contents=[Content(parts=[Part(text=text)])]
        )
        return result.total_tokens

    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        method: Literal["function_calling", "json_mode", "json_schema"]
        | None = "function_calling",
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        parser: OutputParserLike

        # `json_mode` kept for backwards compatibility; shouldn't be used in new code
        if method in ("json_mode", "json_schema"):
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                # Handle Pydantic models
                if issubclass(schema, BaseModelV1):
                    # Use legacy schema generation for pydantic v1 models
                    schema_json = schema.schema()
                else:
                    schema_json = schema.model_json_schema()
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

            llm = self.bind(
                response_mime_type="application/json",
                response_json_schema=schema_json,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema_json,
                },
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
        tool_config: dict | _ToolConfigDict | None = None,
        *,
        tool_choice: _ToolChoiceType | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with google-generativeAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.

                Can be a pydantic model, `Callable`, or `BaseTool`. Pydantic models,
                `Callable`, and `BaseTool` objects will be automatically converted to
                their schema dictionary representation.

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
            formatted_tools: list = [convert_to_openai_tool(tool) for tool in tools]
        except Exception:
            formatted_tools = [
                tool_to_dict(convert_to_genai_function_declarations(tools))
            ]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        elif tool_config:
            kwargs["tool_config"] = tool_config
        else:
            pass
        return self.bind(tools=formatted_tools, **kwargs)

    @property
    def _supports_tool_choice(self) -> bool:
        """Whether the model supports the `tool_choice` parameter.

        See the [Gemini models docs](https://ai.google.dev/gemini-api/docs/models) for a
        full list. Gemini calls this "function calling".
        """
        return (
            "gemini-1.5-pro" in self.model
            or "gemini-1.5-flash" in self.model
            or "gemini-2" in self.model
        )


def _get_tool_name(
    tool: _ToolDict | GoogleTool | dict,
) -> str:
    try:
        genai_tool = tool_to_dict(convert_to_genai_function_declarations([tool]))
        return next(f["name"] for f in genai_tool["function_declarations"])  # type: ignore[index]
    except ValueError:  # other TypedDict
        if is_typeddict(tool):
            return convert_to_openai_tool(cast("dict", tool))["function"]["name"]
        raise
