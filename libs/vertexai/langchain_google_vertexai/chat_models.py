"""Wrapper around Google VertexAI chat-based models."""

from __future__ import annotations  # noqa
import ast
import base64
from functools import cached_property
import json
import logging
import re
from dataclasses import dataclass, field
from operator import itemgetter
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    Literal,
    Tuple,
    TypedDict,
    overload,
)

import proto  # type: ignore[import-untyped]

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
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
from vertexai.generative_models import (  # type: ignore
    Tool as VertexTool,
)
from vertexai.generative_models._generative_models import (  # type: ignore
    ToolConfig,
    SafetySettingsType,
    GenerationConfigType,
    GenerationResponse,
    _convert_schema_dict_to_gapic,
)
from vertexai.language_models import (  # type: ignore
    ChatMessage,
    InputOutputTextPair,
)
from google.cloud.aiplatform_v1.types import (
    Content as v1Content,
    FunctionCallingConfig as v1FunctionCallingConfig,
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
from langchain_google_vertexai._base import _VertexAICommon
from langchain_google_vertexai._image_utils import ImageBytesLoader
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
from typing_extensions import Self, is_typeddict
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
    "thinking_budget",
    "include_thoughts",
]
_allowed_params_prediction_service = ["request", "timeout", "metadata", "labels"]


@dataclass
class _ChatHistory:
    """Represents a context and a history of messages."""

    history: List[ChatMessage] = field(default_factory=list)
    context: Optional[str] = None


class _GeminiGenerateContentKwargs(TypedDict):
    generation_config: Optional[GenerationConfigType]
    safety_settings: Optional[SafetySettingsType]
    tools: Optional[List[VertexTool]]
    tool_config: Optional[ToolConfig]


def _parse_chat_history(history: List[BaseMessage]) -> _ChatHistory:
    """Parse a sequence of messages into history.

    Args:
        history: The list of messages to re-create the history of the chat.
    Returns:
        A parsed chat history.
    Raises:
        ValueError: If a sequence of message has a SystemMessage not at the
        first place.
    """

    vertex_messages, context = [], None
    for i, message in enumerate(history):
        content = cast(str, message.content)
        if i == 0 and isinstance(message, SystemMessage):
            context = content
        elif isinstance(message, AIMessage):
            vertex_message = ChatMessage(content=message.content, author="bot")
            vertex_messages.append(vertex_message)
        elif isinstance(message, HumanMessage):
            vertex_message = ChatMessage(content=message.content, author="user")
            vertex_messages.append(vertex_message)
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )
    chat_history = _ChatHistory(context=context, history=vertex_messages)
    return chat_history


def _parse_chat_history_gemini(
    history: List[BaseMessage],
    imageBytesLoader: ImageBytesLoader,
    perform_literal_eval_on_string_raw_content: Optional[bool] = False,
) -> tuple[Content | None, list[Content]]:
    def _convert_to_prompt(part: Union[str, Dict]) -> Optional[Part]:
        if isinstance(part, str):
            return Part(text=part)

        if not isinstance(part, Dict):
            raise ValueError(
                f"Message's content is expected to be a dict, got {type(part)}!"
            )
        if part["type"] == "text":
            return Part(text=part["text"])
        if part["type"] == "tool_use":
            if part.get("text"):
                return Part(text=part["text"])
            else:
                return None
        if part["type"] == "executable_code":
            if "executable_code" not in part or "language" not in part:
                raise ValueError(
                    "Executable code part must have 'code' and 'language' keys, got "
                    f"{part}"
                )
            return Part(
                executable_code=ExecutableCode(
                    language=part["language"], code=part["executable_code"]
                )
            )
        if part["type"] == "code_execution_result":
            if "code_execution_result" not in part or "outcome" not in part:
                raise ValueError(
                    "Code execution result part must have 'code_execution_result' and "
                    f"'outcome' keys, got {part}"
                )
            return Part(
                code_execution_result=CodeExecutionResult(
                    output=part["code_execution_result"], outcome=part["outcome"]
                )
            )

        if is_data_content_block(part):
            # LangChain standard format
            if part["type"] == "image" and part["source_type"] == "url":
                oai_content_block = convert_to_openai_image_block(part)
                url = oai_content_block["image_url"]["url"]
                return imageBytesLoader.load_gapic_part(url)
            elif part["source_type"] == "base64":
                bytes_ = base64.b64decode(part["data"])
            else:
                raise ValueError("source_type must be url or base64.")
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
                raise ValueError(f"Missing mime_type in media part: {part}")
            mime_type = part["mime_type"]
            proto_part = Part()

            if "data" in part:
                proto_part.inline_data = Blob(data=part["data"], mime_type=mime_type)
            elif "file_uri" in part:
                proto_part.file_data = FileData(
                    file_uri=part["file_uri"], mime_type=mime_type
                )
            else:
                raise ValueError(
                    f"Media part must have either data or file_uri: {part}"
                )

            if "video_metadata" in part:
                metadata = VideoMetadata(part["video_metadata"])
                proto_part.video_metadata = metadata
            return proto_part

        if part["type"] == "thinking":
            return Part(text=part["thinking"], thought=True)

        raise ValueError("Only text, image_url, and media types are supported!")

    def _convert_to_parts(message: BaseMessage) -> List[Part]:
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
        # A linting error is thrown here because it does not think this line is
        # reachable due to typing, but mypy is wrong so we ignore the lint
        # error.
        if isinstance(raw_content, int):  # type: ignore
            raw_content = str(raw_content)  # type: ignore
        if isinstance(raw_content, str):
            raw_content = [raw_content]
        result = []
        for raw_part in raw_content:
            part = _convert_to_prompt(raw_part)
            if part:
                result.append(part)
        return result

    vertex_messages: List[Content] = []
    system_parts: List[Part] | None = None
    system_instruction = None

    # the last AI Message before a sequence of tool calls
    prev_ai_message: Optional[AIMessage] = None

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

            parts = []
            if message.content:
                parts = _convert_to_parts(message)

            for tc in message.tool_calls:
                function_call = FunctionCall({"name": tc["name"], "args": tc["args"]})
                parts.append(Part(function_call=function_call))

            if len(vertex_messages):
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
            role = "function"

            part = Part(
                function_response=FunctionResponse(
                    name=message.name, response={"content": message.content}
                )
            )
            parts = [part]
            if len(vertex_messages):
                prev_content = vertex_messages[-1]
                prev_content_is_function = (
                    prev_content and prev_content.role == "function"
                )
                if prev_content_is_function:
                    prev_parts = list(prev_content.parts)
                    prev_parts.extend(parts)
                    # replacing last message
                    vertex_messages[-1] = Content(role=role, parts=prev_parts)
                    continue

            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, ToolMessage):
            role = "function"

            # message.name can be null for ToolMessage
            name = message.name
            if name is None:
                if prev_ai_message:
                    tool_call_id = message.tool_call_id
                    tool_call: ToolCall | None = next(
                        (
                            t
                            for t in prev_ai_message.tool_calls
                            if t["id"] == tool_call_id
                        ),
                        None,
                    )

                    if tool_call is None:
                        raise ValueError(
                            (
                                "Message name is empty and can't find"
                                + f"corresponding tool call for id: '${tool_call_id}'"
                            )
                        )
                    name = tool_call["name"]

            def _parse_content(raw_content: str | Dict[Any, Any]) -> Dict[Any, Any]:
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
                    merged_content: Dict[Any, Any] = {}
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
            prev_content_is_function = prev_content and prev_content.role == "function"

            if prev_content_is_function:
                prev_parts = list(prev_content.parts)
                prev_parts.extend(parts)
                # replacing last message
                vertex_messages[-1] = Content(role=role, parts=prev_parts)
                continue

            vertex_messages.append(Content(role=role, parts=parts))
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )
    return system_instruction, vertex_messages


def _parse_examples(examples: List[BaseMessage]) -> List[InputOutputTextPair]:
    if len(examples) % 2 != 0:
        raise ValueError(
            f"Expect examples to have an even amount of messages, got {len(examples)}."
        )
    example_pairs = []
    input_text = None
    for i, example in enumerate(examples):
        if i % 2 == 0:
            if not isinstance(example, HumanMessage):
                raise ValueError(
                    f"Expected the first message in a part to be from human, got "
                    f"{type(example)} for the {i}th message."
                )
            input_text = example.content
        if i % 2 == 1:
            if not isinstance(example, AIMessage):
                raise ValueError(
                    f"Expected the second message in a part to be from AI, got "
                    f"{type(example)} for the {i}th message."
                )
            pair = InputOutputTextPair(
                input_text=input_text, output_text=example.content
            )
            example_pairs.append(pair)
    return example_pairs


def _get_question(messages: List[BaseMessage]) -> HumanMessage:
    """Get the human message at the end of a list of input messages to a chat model."""
    if not messages:
        raise ValueError("You should provide at least one message to start the chat!")
    question = messages[-1]
    if not isinstance(question, HumanMessage):
        raise ValueError(
            f"Last message in the list should be from human, got {question.type}."
        )
    return question


@overload
def _parse_response_candidate(
    response_candidate: "Candidate", streaming: Literal[False] = False
) -> AIMessage:
    ...


@overload
def _parse_response_candidate(
    response_candidate: "Candidate", streaming: Literal[True]
) -> AIMessageChunk:
    ...


def _parse_response_candidate(
    response_candidate: "Candidate", streaming: bool = False
) -> AIMessage:
    content: Union[None, str, List[Union[str, dict[str, Any]]]] = None
    additional_kwargs = {}
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    for part in response_candidate.content.parts:
        try:
            text: Optional[str] = part.text
        except AttributeError:
            text = None

        if part.thought:
            thinking_message = {
                "type": "thinking",
                "thinking": part.text,
            }
            if not content:
                content = [thinking_message]
            elif isinstance(content, str):
                content = [thinking_message, content]
            elif isinstance(content, list):
                content.append(thinking_message)
            else:
                raise Exception("Unexpected content type")

        elif text:
            if not content:
                content = text
            elif isinstance(content, str):
                content = [content, text]
            elif isinstance(content, list):
                content.append(text)
            else:
                raise Exception("Unexpected content type")

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

            if streaming:
                index = function_call.get("index")
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=function_call.get("id", str(uuid.uuid4())),
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
                                id=tool_call.get("id", str(uuid.uuid4())),
                            )
                            for tool_call in tool_calls_dicts
                        ]
                    )
                except Exception as e:
                    invalid_tool_calls.append(
                        invalid_tool_call(
                            name=function_call.get("name"),
                            args=function_call.get("arguments"),
                            id=function_call.get("id", str(uuid.uuid4())),
                            error=str(e),
                        )
                    )
        if hasattr(part, "executable_code") and part.executable_code is not None:
            if part.executable_code.code and part.executable_code.language:
                code_message = {
                    "type": "executable_code",
                    "executable_code": part.executable_code.code,
                    "language": part.executable_code.language,
                }
                if not content:
                    content = [code_message]
                elif isinstance(content, str):
                    content = [content, code_message]
                elif isinstance(content, list):
                    content.append(code_message)
                else:
                    raise Exception("Unexpected content type")

        if (
            hasattr(part, "code_execution_result")
            and part.code_execution_result is not None
        ):
            if part.code_execution_result.output and part.code_execution_result.outcome:
                execution_result = {
                    "type": "code_execution_result",
                    # Name output -> code_execution_result for consistency with
                    # langchain-google-genai
                    "code_execution_result": part.code_execution_result.output,
                    "outcome": part.code_execution_result.outcome,
                }

                if not content:
                    content = [execution_result]
                elif isinstance(content, str):
                    content = [content, execution_result]
                elif isinstance(content, list):
                    content.append(execution_result)
                else:
                    raise Exception("Unexpected content type")

    if content is None:
        content = ""

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
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    wait_exponential_kwargs: Optional[dict[str, float]] = None,
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
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    wait_exponential_kwargs: Optional[dict[str, float]] = None,
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

    params = {
        k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service
    }
    return await _completion_with_retry_inner(
        generation_method,
        **params,
    )


class ChatVertexAI(_VertexAICommon, BaseChatModel):
    """Google Cloud Vertex AI chat model integration.

    Setup:
        You must either:
            - Have credentials configured for your environment (gcloud, workload identity, etc...)
            - Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

        This codebase uses the google.auth library which first looks for the application
        credentials variable mentioned above, and then looks for system-level auth.

        For more information, see:
        https://cloud.google.com/docs/authentication/application-default-credentials#GAC
        and https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth.

    Key init args — completion params:
        model: str
            Name of ChatVertexAI model to use. e.g. "gemini-2.0-flash-001",
            "gemini-2.5-pro", etc.
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
            - multiplier: Initial wait time multiplier (default: 1.0)
            - min: Minimum wait time in seconds (default: 4.0)
            - max: Maximum wait time in seconds (default: 10.0)
            - exp_base: Exponent base to use (default: 2.0)
        credentials: Optional[google.auth.credentials.Credentials]
            The default custom credentials to use when making API calls. If not
            provided, credentials will be ascertained from the environment.
        project: Optional[str]
            The default GCP project to use when making Vertex API calls.
        location: str = "us-central1"
            The default location to use when making API calls.
        request_parallelism: int = 5
            The amount of parallelism allowed for requests issued to VertexAI models.
            Default is 5.
        base_url: Optional[str]
            Base URL for API requests.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(
                model="gemini-1.5-flash-001",
                temperature=0,
                max_tokens=None,
                max_retries=6,
                stop=None,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content="J'adore programmer. ", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-925ce305-2268-44c4-875f-dde9128520ad-0')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='J', response_metadata={'is_blocked': False, 'safety_ratings': [], 'citation_metadata': None}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content="'adore programmer. ", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}], 'citation_metadata': None}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content='', response_metadata={'is_blocked': False, 'safety_ratings': [], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'adore programmer. ", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-b7f7492c-4cb5-42d0-8fc3-dce9b293b0fb')

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(content="J'adore programmer. ", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'probability_score': 0.1, 'blocked': False, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severity_score': 0.1}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-925ce305-2268-44c4-875f-dde9128520ad-0')

    Context Caching:
        Context caching allows you to store and reuse content (e.g., PDFs, images) for faster processing.
        The `cached_content` parameter accepts a cache name created via the Google Generative AI API with Vertex AI.
        Below is an example of caching content from GCS and querying it.

        Example:
        This caches content from GCS and queries it.

        .. code-block:: python

            from google import genai
            from google.genai.types import Content, CreateCachedContentConfig, HttpOptions, Part
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
                model="gemini-1.5-flash-001",
                config=CreateCachedContentConfig(
                    contents=contents,
                    system_instruction="You are an expert content analyzer.",
                    display_name="content-cache",
                    ttl="300s",
                ),
            )

            llm = ChatVertexAI(
                model_name="gemini-1.5-flash-001",
                cached_content=cache.name,
            )
            message = HumanMessage(content="Provide a summary of the key information across the content.")
            llm.invoke([message])

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': '2a2401fa-40db-470d-83ce-4e52de910d9e'},
             {'name': 'GetWeather',
              'args': {'location': 'New York City, NY'},
              'id': '96761deb-ab7f-4ef9-b4b4-6d44562fc46e'},
             {'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': '9147d532-abee-43a2-adb5-12f164300484'},
             {'name': 'GetPopulation',
              'args': {'location': 'New York City, NY'},
              'id': 'c43374ea-bde5-49ca-8487-5b83ebeea1e6'}]

        See ``ChatVertexAI.bind_tools()`` method for more.

    Built-in search:
        .. code-block:: python

            from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(model="gemini-2.0-flash-exp")
            resp = llm.invoke(
                "When is the next total solar eclipse in US?",
                tools=[VertexTool(google_search={})],
            )

    Built-in code execution:
        .. code-block:: python

            from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(model="gemini-2.0-flash-exp")
            resp = llm.invoke(
                "What is 3^3?",
                tools=[VertexTool(code_execution={})],
            )

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(default=None, description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup='What do you call a cat that loves to bowl?', punchline='An alley cat!', rating=None)

        See ``ChatVertexAI.with_structured_output()`` for more.

    Image input:
        .. code-block:: python

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

        .. code-block:: python

            'The weather in this image appears to be sunny and pleasant. The sky is a bright blue with scattered white clouds, suggesting a clear and mild day. The lush green grass indicates recent rainfall or sufficient moisture. The absence of strong shadows suggests that the sun is high in the sky, possibly late afternoon. Overall, the image conveys a sense of tranquility and warmth, characteristic of a beautiful summer day.'

        You can also point to GCS files which is faster / more efficient because bytes are transferred back and forth.

        .. code-block:: python

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

        .. code-block:: python

            'The image is of five blueberry scones arranged on a piece of baking paper. Here is a list of what is in the picture:* **Five blueberry scones:** They are scattered across the parchment paper, dusted with powdered sugar.  * **Two cups of coffee:**  Two white cups with saucers. One appears full, the other partially drunk. * **A bowl of blueberries:** A brown bowl is filled with fresh blueberries, placed near the scones.* **A spoon:**  A silver spoon with the words "Let\'s Jam" rests on the paper.* **Pink peonies:** Several pink peonies lie beside the scones, adding a touch of color.* **Baking paper:** The scones, cups, bowl, and spoon are arranged on a piece of white baking paper, splattered with purple.  The paper is crinkled and sits on a dark surface. The image has a rustic and delicious feel, suggesting a cozy and enjoyable breakfast or brunch setting.' # codespell:ignore brunch

    Video input:
        **NOTE**: Currently only supported for ``gemini-...-vision`` models.

        .. code-block:: python

            llm = ChatVertexAI(model="gemini-1.0-pro-vision")

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

        .. code-block:: python

             'The video is about a new feature in Google Photos called "Zoomable Selfies". The feature allows users to take selfies with animals at the zoo. The video shows several examples of people taking selfies with animals, including a tiger, an elephant, and a sea otter. The video also shows how the feature works. Users simply need to open the Google Photos app and select the "Zoomable Selfies" option. Then, they need to choose an animal from the list of available animals. The app will then guide the user through the process of taking the selfie.'

    Audio input:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            llm = ChatVertexAI(model="gemini-1.5-flash-001")

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

        .. code-block:: python

            "This audio is an interview with two product managers from Google who work on Pixel feature drops. They discuss how feature drops are important for showcasing how Google devices are constantly improving and getting better. They also discuss some of the highlights of the January feature drop and the new features coming in the March drop for Pixel phones and Pixel watches. The interview concludes with discussion of how user feedback is extremely important to them in deciding which features to include in the feature drops. "

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 17, 'output_tokens': 7, 'total_tokens': 24}

    Logprobs:
        .. code-block:: python

            llm = ChatVertexAI(model="gemini-1.5-flash-001", logprobs=True)
            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata["logprobs_result"]

        .. code-block:: python

            [
                {'token': 'J', 'logprob': -1.549651415189146e-06, 'top_logprobs': []},
                {'token': "'", 'logprob': -1.549651415189146e-06, 'top_logprobs': []},
                {'token': 'adore', 'logprob': 0.0, 'top_logprobs': []},
                {'token': ' programmer', 'logprob': -1.1922384146600962e-07, 'top_logprobs': []},
                {'token': '.', 'logprob': -4.827636439586058e-05, 'top_logprobs': []},
                {'token': ' ', 'logprob': -0.018011733889579773, 'top_logprobs': []},
                {'token': '\\n', 'logprob': -0.0008687592926435173, 'top_logprobs': []}
            ]



    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'is_blocked': False,
             'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_HARASSMENT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1}],
             'usage_metadata': {'prompt_token_count': 17,
              'candidates_token_count': 7,
              'total_token_count': 24}}

    Safety settings
        .. code-block:: python

            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

            llm = ChatVertexAI(
                model="gemini-1.5-pro",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                },
            )

            llm.invoke(messages).response_metadata

        .. code-block:: python

            {'is_blocked': False,
             'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_HARASSMENT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1},
              {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
               'probability_label': 'NEGLIGIBLE',
               'probability_score': 0.1,
               'blocked': False,
               'severity': 'HARM_SEVERITY_NEGLIGIBLE',
               'severity_score': 0.1}],
             'usage_metadata': {'prompt_token_count': 17,
              'candidates_token_count': 7,
              'total_token_count': 24}}

    """  # noqa: E501

    model_name: str = Field(alias="model")
    "Underlying model name."
    examples: Optional[List[BaseMessage]] = None
    response_mime_type: Optional[str] = None
    """Optional. Output response mimetype of the generated candidate text. Only
        supported in Gemini 1.5 and later models. Supported mimetype:
            * "text/plain": (default) Text output.
            * "application/json": JSON response in the candidates.
            * "text/x.enum": Enum in plain text.
       The model also needs to be prompted to output the appropriate response
       type, otherwise the behavior is undefined. This is a preview feature.
    """

    response_schema: Optional[Dict[str, Any]] = None
    """ Optional. Enforce an schema to the output.
        The format of the dictionary should follow Open API schema.
    """

    cached_content: Optional[str] = None
    """ Optional. Use the model in cache mode. Only supported in Gemini 1.5 and later
        models. Must be a string containing the cache name (A sequence of numbers)
    """

    logprobs: Union[bool, int] = False
    """Whether to return logprobs as part of AIMessage.response_metadata.

    If False, don't return logprobs. If True, return logprobs for top candidate.
    If int, return logprobs for top ``logprobs`` candidates.

    **NOTE**: As of 10.28.24 this is only supported for gemini-1.5-flash models.

    .. versionadded: 2.0.6
    """
    labels: Optional[Dict[str, str]] = None
    """ Optional tag llm calls with metadata to help in tracebility and biling.
    """

    perform_literal_eval_on_string_raw_content: bool = True
    """Whether to perform literal eval on string raw content.
    """

    wait_exponential_kwargs: Optional[dict[str, float]] = None
    """Optional dictionary with parameters for wait_exponential:
        - multiplier: Initial wait time multiplier (default: 1.0)
        - min: Minimum wait time in seconds (default: 4.0)
        - max: Maximum wait time in seconds (default: 10.0)
        - exp_base: Exponent base to use (default: 2.0)
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any unexpected initialization parameters."""

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg
        and for arg validation.
        """
        if model_name:
            kwargs["model_name"] = model_name

        # Get all valid field names, including aliases
        valid_fields = set()
        for field_name, field_info in self.model_fields.items():
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
                    f"provided to ChatVertexAI.{suggestion}"
                )
        super().__init__(**kwargs)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "vertexai"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if self.labels:
            for key, value in self.labels.items():
                if not re.match(r"^[a-z][a-z0-9-_]{0,62}$", key):
                    raise ValueError(f"Invalid label key: {key}")
                if value and len(value) > 63:
                    raise ValueError(f"Label value too long: {value}")
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
                project=cast(str, self.project),
            )
        else:
            self.full_model_name = _format_model_name(
                self.model_name,
                location=self.location,
                project=cast(str, self.project),
            )

        return self

    def _prepare_params(
        self,
        stop: Optional[List[str]] = None,
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
        _ = params.pop("thinking_budget", None)

        include_thoughts = kwargs.get("include_thoughts", self.include_thoughts)
        if include_thoughts is not None:
            if "thinking_config" not in params:
                params["thinking_config"] = {}
            params["thinking_config"]["include_thoughts"] = include_thoughts
        _ = params.pop("include_thoughts", None)

        return params

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._prepare_params(stop=stop, **kwargs)
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.
            stream: Whether to use the streaming endpoint.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
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
        stop: Optional[List[str]] = None,
        stream: bool = False,
        *,
        logprobs: int | bool = False,
        **kwargs: Any,
    ) -> Union[GenerationConfig, v1GenerationConfig]:
        """Prepares GenerationConfig part of the request.

        https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#generationconfig
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
            )
        )

    def _safety_settings_gemini(
        self, safety_settings: Optional[SafetySettingsType]
    ) -> Optional[Sequence[SafetySetting]]:
        """Prepares SafetySetting part of the request.

        https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#safetysetting
        """
        if safety_settings is None:
            if self.safety_settings:
                return self._safety_settings_gemini(self.safety_settings)
            return None
        if isinstance(safety_settings, list):
            return safety_settings
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
        raise ValueError("safety_settings should be either")

    def _prepare_request_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[_ToolsType] = None,
        functions: Optional[_ToolsType] = None,
        tool_config: Optional[Union[_ToolConfigDict, ToolConfig]] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        cached_content: Optional[str] = None,
        *,
        tool_choice: Optional[_ToolChoiceType] = None,
        logprobs: Optional[Union[int, bool]] = None,
        **kwargs,
    ) -> Union[v1GenerateContentRequest, GenerateContentRequest]:
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
        safety_settings = self._safety_settings_gemini(safety_settings)
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
                        **proto.Message.to_dict(tool_config.function_calling_config)
                    )
                )

            if safety_settings:
                v1_safety_settings = [
                    v1SafetySetting(
                        category=s.category, method=s.method, threshold=s.threshold
                    )
                    for s in safety_settings
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
                safety_settings=safety_settings,
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
            safety_settings=safety_settings,
            generation_config=generation_config,
            model=self.full_model_name,
            labels=self.labels,
        )

    def _request_from_cached_content(
        self,
        cached_content: str,
        system_instruction: Optional[Content],
        tools: Optional[Sequence[GapicTool]],
        tool_config: Optional[Union[_ToolConfigDict, ToolConfig]],
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)
        response = _completion_with_retry(
            self.prediction_client.generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            **kwargs,
        )
        return self._gemini_response_to_chat_result(response)

    async def _agenerate_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = await _acompletion_with_retry(
            self.async_prediction_client.generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=self._prepare_request_gemini(
                messages=messages, stop=stop, **kwargs
            ),
            metadata=self.default_metadata,
            **kwargs,
        )
        return self._gemini_response_to_chat_result(response)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        # https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#counttokensrequest
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

    def _tools_gemini(
        self,
        tools: Optional[_ToolsType] = None,
        functions: Optional[_ToolsType] = None,
    ) -> Optional[List[GapicTool]]:
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
        self, tool_config: Optional[Union[_ToolConfigDict, ToolConfig]] = None
    ) -> Optional[GapicToolConfig]:
        if tool_config and not isinstance(tool_config, ToolConfig):
            return _format_tool_config(cast(_ToolConfigDict, tool_config))
        return None

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.

        Returns:
            The ChatResult that contains outputs generated by the model.

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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield from self._stream_gemini(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return

    def _stream_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)
        response_iter = _completion_with_retry(
            self.prediction_client.stream_generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            **kwargs,
        )
        total_lc_usage = None
        for response_chunk in response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                run_manager.on_llm_new_token(chunk.message.content)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # TODO: Update to properly support async streaming from gemini.
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)

        response_iter = _acompletion_with_retry(
            self.async_prediction_client.stream_generate_content,
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
            request=request,
            metadata=self.default_metadata,
            **kwargs,
        )
        total_lc_usage = None
        async for response_chunk in await response_iter:
            chunk, total_lc_usage = self._gemini_chunk_to_generation_chunk(
                response_chunk, prev_total_usage=total_lc_usage
            )
            if run_manager and isinstance(chunk.message.content, str):
                await run_manager.on_llm_new_token(chunk.message.content)
            yield chunk

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel], Type],
        *,
        include_raw: bool = False,
        method: Optional[Literal["json_mode"]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        .. versionchanged:: 1.1.0

            Return type corrected in version 1.1.0. Previously if a dict schema
            was provided then the output had the form
            ``[{"args": {}, "name": "schema_name"}]`` where the output was a list with
            a single dict and the "args" of the one dict corresponded to the schema.
            As of `1.1.0` this has been fixed so that the schema (the value
            corresponding to the old "args" key) is returned directly.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".
            method: If set to 'json_schema' it will use controlled genetration to
                generate the response rather than function calling. Does not work with
                schemas with references or Pydantic models with self-references.

        Returns:
            A Runnable that takes any ChatModel input. If include_raw is True then a
            dict with keys — raw: BaseMessage, parsed: Optional[_DictOrPydantic],
            parsing_error: Optional[BaseException]. If include_raw is False then just
            _DictOrPydantic is returned, where _DictOrPydantic depends on the schema.
            If schema is a Pydantic class then _DictOrPydantic is the Pydantic class.
            If schema is a dict then _DictOrPydantic is a dict.

        Example: Pydantic schema, exclude raw:
            .. code-block:: python

                from pydantic import BaseModel
                from langchain_google_vertexai import ChatVertexAI

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> AnswerWithJustification(
                #     answer='They weigh the same.', justification='A pound is a pound.'
                # )

        Example: Pydantic schema, include raw:
            .. code-block:: python

                from pydantic import BaseModel
                from langchain_google_vertexai import ChatVertexAI

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Dict schema, exclude raw:
            .. code-block:: python

                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_function
                from langchain_google_vertexai import ChatVertexAI

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_function(AnswerWithJustification)
                llm = ChatVertexAI(model_name="gemini-2.0-flash-001", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: Pydantic schema, streaming:
            .. code-block:: python

                from pydantic import BaseModel, Field
                from langchain_google_vertexai import ChatVertexAI

                class Explanation(BaseModel):
                    '''A topic explanation with examples.'''
                    description: str = Field(description="A brief description of the topic.")
                    examples: str = Field(description="Two examples related to the topic.")

                llm = ChatVertexAI(model_name="gemini-2.0-flash", temperature=0)
                structured_llm = llm.with_structured_output(Explanation, method="json_mode")

                for chunk in structured_llm.stream("Tell me about transformer models"):
                    print(chunk)
                    print('-------------------------')
                # -> description='Transformer models are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) and are also increasingly used in computer vision and other domains. They rely on the self-attention mechanism to weigh the importance of different parts of the input data, allowing them to effectively capture long-range dependencies. Unlike recurrent neural networks (RNNs), transformers can process the entire input sequence in parallel, leading to significantly faster training times. Key components of transformer models include: the self-attention mechanism (calculates attention weights between different parts of the input), multi-head attention (performs self-attention multiple times with different learned parameters), positional encoding (adds information about the position of tokens in the input sequence), feedforward networks (applies a non-linear transformation to each position), and encoder-decoder structure (used for sequence-to-sequence tasks).' examples='1. BERT (Bidirectional Encoder Representations from Transformers): A pre-trained transformer'
                #    -------------------------
                #    description='Transformer models are a type of neural network architecture that have revolutionized the field of natural language processing (NLP) and are also increasingly used in computer vision and other domains. They rely on the self-attention mechanism to weigh the importance of different parts of the input data, allowing them to effectively capture long-range dependencies. Unlike recurrent neural networks (RNNs), transformers can process the entire input sequence in parallel, leading to significantly faster training times. Key components of transformer models include: the self-attention mechanism (calculates attention weights between different parts of the input), multi-head attention (performs self-attention multiple times with different learned parameters), positional encoding (adds information about the position of tokens in the input sequence), feedforward networks (applies a non-linear transformation to each position), and encoder-decoder structure (used for sequence-to-sequence tasks).' examples='1. BERT (Bidirectional Encoder Representations from Transformers): A pre-trained transformer model that can be fine-tuned for various NLP tasks like text classification, question answering, and named entity recognition. 2. GPT (Generative Pre-trained Transformer): A language model that uses transformers to generate coherent and contextually relevant text. GPT models are used in chatbots, content creation, and code generation.'
                #    -------------------------

        """  # noqa: E501

        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        parser: OutputParserLike

        if method == "json_mode":
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                if issubclass(schema, BaseModelV1):
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
                    raise ValueError(f"Unsupported schema type {type(schema)}")
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
                raise ValueError(f"Unsupported schema type {type(schema)}")
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
        else:
            return llm | parser

    def bind_tools(
        self,
        tools: _ToolsType,
        tool_config: Optional[_ToolConfigDict] = None,
        *,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Vertex tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        if tool_choice and tool_config:
            raise ValueError(
                "Must specify at most one of tool_choice and tool_config, received "
                f"both:\n\n{tool_choice=}\n\n{tool_config=}"
            )
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
            message.response_metadata["model_name"] = self.model_name
            if isinstance(message, AIMessage):
                message.usage_metadata = lc_usage
            generations.append(ChatGeneration(message=message, generation_info=info))
        if not response.candidates:
            message = AIMessage(content="")
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
        prev_total_usage: Optional[UsageMetadata] = None,
    ) -> Tuple[ChatGenerationChunk, Optional[UsageMetadata]]:
        # return an empty completion message if there's no candidates
        usage_metadata = proto.Message.to_dict(response_chunk.usage_metadata)

        # Gather langchain (standard) usage metadata
        # Note: some models (e.g., gemini-1.5-pro with image inputs) return
        # cumulative sums of token counts.
        total_lc_usage = _get_usage_metadata_gemini(usage_metadata)
        if total_lc_usage and prev_total_usage:
            lc_usage: Optional[UsageMetadata] = UsageMetadata(
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
            # add model name if final chunk
            if generation_info.get("finish_reason"):
                message.response_metadata["model_name"] = self.model_name
            # is_blocked is part of "safety_ratings" list
            # but if it's True/False then chunks can't be marged
            generation_info.pop("is_blocked", None)
        return ChatGenerationChunk(
            message=message,
            generation_info=generation_info,
        ), total_lc_usage


def _get_usage_metadata_gemini(raw_metadata: dict) -> Optional[UsageMetadata]:
    """Get UsageMetadata from raw response metadata."""
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
    else:
        if thought_tokens > 0:
            return UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_token_details={"cache_read": cache_read_tokens},
                output_token_details={"reasoning": thought_tokens},
            )
        else:
            return UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_token_details={"cache_read": cache_read_tokens},
            )


def _get_tool_name(tool: _ToolType) -> str:
    vertexai_tool = _format_to_gapic_tool([tool])
    return [f.name for f in vertexai_tool.function_declarations][0]
