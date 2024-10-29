"""Wrapper around Google VertexAI chat-based models."""

from __future__ import annotations  # noqa
import ast
import json
import logging
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
from google.cloud.aiplatform import telemetry

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
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
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
    ChatModel,
    ChatSession,
    CodeChatModel,
    CodeChatSession,
    InputOutputTextPair,
)
from vertexai.preview.language_models import (  # type: ignore
    ChatModel as PreviewChatModel,
)
from vertexai.preview.language_models import (
    CodeChatModel as PreviewCodeChatModel,
)

from google.cloud.aiplatform_v1beta1.types import (
    Blob,
    Candidate,
    Part,
    HarmCategory,
    Content,
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
from langchain_google_vertexai._base import _VertexAICommon, GoogleModelFamily
from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
    _format_model_name,
    is_gemini_model,
    replace_defs_in_schema,
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
from typing_extensions import Self


logger = logging.getLogger(__name__)


_allowed_params = [
    "temperature",
    "top_k",
    "top_p",
    "response_mime_type",
    "response_schema",
    "temperature",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "candidate_count",
    "seed",
    "response_logprobs",
    "logprobs",
]
_allowed_params_prediction_service = ["request", "timeout", "metadata"]


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
    project: Optional[str] = None,
    convert_system_message_to_human: Optional[bool] = False,
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
        if part["type"] == "image_url":
            path = part["image_url"]["url"]
            return ImageBytesLoader(project=project).load_gapic_part(path)

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

        raise ValueError("Only text, image_url, and media types are supported!")

    def _convert_to_parts(message: BaseMessage) -> List[Part]:
        raw_content = message.content

        # If a user sends a multimodal request with agents, then the full input
        # will be sent as a string due to the ChatPromptTemplate formatting.
        # Because of this, we need to first try to convert the string to its
        # native type (such as list or dict) so that results can be properly
        # appended to the prompt, otherwise they will all be parsed as Text
        # rather than `inline_data`.
        if isinstance(raw_content, str):
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
            if convert_system_message_to_human:
                logger.warning(
                    "gemini models released from April 2024 support"
                    "SystemMessages natively. For best performances,"
                    "when working with these models,"
                    "set convert_system_message_to_human to False"
                )
                continue
            if system_instruction is not None:
                system_instruction.parts.extend(system_parts)  # type: ignore[unreachable]
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
                else:
                    content = parsed_content[0]
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
    content: Union[None, str, List[str]] = None
    additional_kwargs = {}
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    for part in response_candidate.content.parts:
        try:
            text: Optional[str] = part.text
        except AttributeError:
            text = None

        if text:
            if not content:
                content = text
            elif isinstance(content, str):
                content = [content, text]
            elif isinstance(content, list):
                content.append(text)
            else:
                raise Exception("Unexpected content type")

        if part.function_call:
            if "function_call" in additional_kwargs:
                logger.warning(
                    (
                        "This model can reply with multiple "
                        "function calls in one response. "
                        "Please don't rely on `additional_kwargs.function_call` "
                        "as only the last one will be saved."
                        "Use `tool_calls` instead."
                    )
                )
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
    if content is None:
        content = ""

    if streaming:
        return AIMessageChunk(
            content=cast(Union[str, List[Union[str, Dict[Any, Any]]]], content),
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=cast(Union[str, List[Union[str, Dict[Any, Any]]]], content),
        tool_calls=tool_calls,
        additional_kwargs=additional_kwargs,
        invalid_tool_calls=invalid_tool_calls,
    )


def _completion_with_retry(
    generation_method: Callable,
    *,
    max_retries: int,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry_inner(generation_method: Callable, **kwargs: Any) -> Any:
        return generation_method(**kwargs)

    params = (
        {k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service}
        if kwargs.get("is_gemini")
        else kwargs
    )
    return _completion_with_retry_inner(
        generation_method,
        **params,
    )


async def _acompletion_with_retry(
    generation_method: Callable,
    *,
    max_retries: int,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=max_retries, run_manager=run_manager
    )

    @retry_decorator
    async def _completion_with_retry_inner(
        generation_method: Callable, **kwargs: Any
    ) -> Any:
        return await generation_method(**kwargs)

    params = (
        {k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service}
        if kwargs.get("is_gemini")
        else kwargs
    )
    return await _completion_with_retry_inner(
        generation_method,
        **params,
    )


class ChatVertexAI(_VertexAICommon, BaseChatModel):
    """Google Cloud Vertex AI chat model integration.

    Setup:
        You must have the langchain-google-vertexai Python package installed
        .. code-block:: bash

            pip install -U langchain-google-vertexai

        And either:
            - Have credentials configured for your environment (gcloud, workload identity, etc...)
            - Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

        This codebase uses the google.auth library which first looks for the application
        credentials variable mentioned above, and then looks for system-level auth.

        For more information, see:
        https://cloud.google.com/docs/authentication/application-default-credentials#GAC
        and https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth.

    Key init args — completion params:
        model: str
            Name of ChatVertexAI model to use. e.g. "gemini-1.5-flash-001",
            "gemini-1.5-pro-001", etc.
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

            AIMessage(content="J'adore programmer. \n", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-925ce305-2268-44c4-875f-dde9128520ad-0')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='J', response_metadata={'is_blocked': False, 'safety_ratings': [], 'citation_metadata': None}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content="'adore programmer. \n", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'citation_metadata': None}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content='', response_metadata={'is_blocked': False, 'safety_ratings': [], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'adore programmer. \n", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-b7f7492c-4cb5-42d0-8fc3-dce9b293b0fb')

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(content="J'adore programmer. \n", response_metadata={'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'citation_metadata': None, 'usage_metadata': {'prompt_token_count': 17, 'candidates_token_count': 7, 'total_token_count': 24}}, id='run-925ce305-2268-44c4-875f-dde9128520ad-0')

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

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

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

            'The weather in this image appears to be sunny and pleasant. The sky is a bright blue with scattered white clouds, suggesting a clear and mild day. The lush green grass indicates recent rainfall or sufficient moisture. The absence of strong shadows suggests that the sun is high in the sky, possibly late afternoon. Overall, the image conveys a sense of tranquility and warmth, characteristic of a beautiful summer day. \n'

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

            'The image is of five blueberry scones arranged on a piece of baking paper. \n\nHere is a list of what is in the picture:\n* **Five blueberry scones:** They are scattered across the parchment paper, dusted with powdered sugar.  \n* **Two cups of coffee:**  Two white cups with saucers. One appears full, the other partially drunk.\n* **A bowl of blueberries:** A brown bowl is filled with fresh blueberries, placed near the scones.\n* **A spoon:**  A silver spoon with the words "Let\'s Jam" rests on the paper.\n* **Pink peonies:** Several pink peonies lie beside the scones, adding a touch of color.\n* **Baking paper:** The scones, cups, bowl, and spoon are arranged on a piece of white baking paper, splattered with purple.  The paper is crinkled and sits on a dark surface. \n\nThe image has a rustic and delicious feel, suggesting a cozy and enjoyable breakfast or brunch setting. \n'

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

            {
                'chosen_candidates': [
                    {'token': 'J', 'log_probability': 0.0},
                    {'token': "'", 'log_probability': -4.0052048e-05},
                    {'token': 'adore', 'log_probability': -0.003931577},
                    {'token': ' programmer', 'log_probability': -0.13637993},
                    {'token': '.', 'log_probability': -7.630326e-06},
                    {'token': ' ', 'log_probability': -3.1950593e-05},
                    {'token': '\n', 'log_probability': 0.0}
                ],
                'top_candidates': []
            }


    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'is_blocked': False,
             'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False},
              {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False},
              {'category': 'HARM_CATEGORY_HARASSMENT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False},
              {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False}],
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
               'blocked': False},
              {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False},
              {'category': 'HARM_CATEGORY_HARASSMENT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False},
              {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
               'probability_label': 'NEGLIGIBLE',
               'blocked': False}],
             'usage_metadata': {'prompt_token_count': 17,
              'candidates_token_count': 7,
              'total_token_count': 24}}

    """  # noqa: E501

    model_name: str = Field(default="chat-bison-default", alias="model")
    "Underlying model name."
    examples: Optional[List[BaseMessage]] = None
    convert_system_message_to_human: bool = False
    """[Deprecated] Since new Gemini models support setting a System Message,
    setting this parameter to True is discouraged.
    """
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
    
    If False, don't return logprobs. If True, return logprobs for top candidate. If 
    int, return logprobs for top ``logprobs`` candidates.
    
    **NOTE**: As of 10.28.24 this is only supported for gemini-1.5-flash models.
    
    .. versionadded: 2.0.6
    """

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg."""
        if model_name:
            kwargs["model_name"] = model_name
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

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the python package exists in environment."""
        safety_settings = self.safety_settings
        tuned_model_name = self.tuned_model_name
        self.model_family = GoogleModelFamily(self.model_name)

        if self.model_name == "chat-bison-default":
            logger.warning(
                "Model_name will become a required arg for VertexAIEmbeddings "
                "starting from Sep-01-2024. Currently the default is set to "
                "chat-bison"
            )
            self.model_name = "chat-bison"

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

        if safety_settings and not is_gemini_model(self.model_family):
            raise ValueError("Safety settings are only supported for Gemini models")

        if tuned_model_name:
            generative_model_name = self.tuned_model_name
        else:
            generative_model_name = self.model_name

        if not is_gemini_model(self.model_family):
            logger.warning(
                "Non-Gemini models are deprecated. "
                "They will be remoced starting from Dec-01-2024. "
            )
            values = {
                "project": self.project,
                "location": self.location,
                "credentials": self.credentials,
                "api_transport": self.api_transport,
                "api_endpoint": self.api_endpoint,
                "default_metadata": self.default_metadata,
            }
            self._init_vertexai(values)
            if self.model_family == GoogleModelFamily.CODEY:
                model_cls = CodeChatModel
                model_cls_preview = PreviewCodeChatModel
            else:
                model_cls = ChatModel
                model_cls_preview = PreviewChatModel
            self.client = model_cls.from_pretrained(generative_model_name)
            self.client_preview = model_cls_preview.from_pretrained(
                generative_model_name
            )
        return self

    @property
    def _is_gemini_advanced(self) -> bool:
        return self.model_family == GoogleModelFamily.GEMINI_ADVANCED

    @property
    def _default_params(self) -> Dict[str, Any]:
        updated_params = super()._default_params

        if self.response_mime_type is not None:
            updated_params["response_mime_type"] = self.response_mime_type

        if self.response_schema is not None:
            allowed_mime_types = ("application/json", "text/x.enum")
            if self.response_mime_type not in allowed_mime_types:
                error_message = (
                    "`response_schema` is only supported when "
                    f"`response_mime_type` is set to one of {allowed_mime_types}"
                )
                raise ValueError(error_message)

            gapic_response_schema = _convert_schema_dict_to_gapic(self.response_schema)
            updated_params["response_schema"] = gapic_response_schema

        return updated_params

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
        if not self._is_gemini_model:
            return self._generate_non_gemini(messages, stop=stop, **kwargs)
        return self._generate_gemini(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            is_gemini=True,
            **kwargs,
        )

    def _generation_config_gemini(
        self,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        *,
        logprobs: int | bool = False,
        **kwargs: Any,
    ) -> GenerationConfig:
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
    ) -> GenerateContentRequest:
        system_instruction, contents = _parse_chat_history_gemini(messages)
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
        generation_config = self._generation_config_gemini(
            stream=stream, stop=stop, logprobs=logprobs, **kwargs
        )

        if (self.cached_content is not None) or (cached_content is not None):
            selected_cached_content = self.cached_content or cached_content

            return self._request_from_cached_content(
                cached_content=selected_cached_content,  # type: ignore
                contents=contents,
                system_instruction=system_instruction,
                tools=formatted_tools,
                tool_config=tool_config,
                safety_settings=safety_settings,
                generation_config=generation_config,
                model=self.full_model_name,
            )

        return GenerateContentRequest(
            contents=contents,
            system_instruction=system_instruction,
            tools=formatted_tools,
            tool_config=tool_config,
            safety_settings=safety_settings,
            generation_config=generation_config,
            model=self.full_model_name,
        )

    def _request_from_cached_content(
        self,
        cached_content: str,
        system_instruction: Optional[Content],
        tools: Optional[Sequence[GapicTool]],
        tool_config: Optional[Union[_ToolConfigDict, ToolConfig]],
        contents: list[Content],
        safety_settings: Optional[Sequence[SafetySetting]],
        generation_config: GenerationConfig,
        model: Optional[str],
    ) -> GenerateContentRequest:
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

        full_cache_name = (
            f"projects/{self.project}/locations/{self.location}/"
            f"cachedContents/{cached_content}"
        )

        return GenerateContentRequest(
            contents=contents,
            model=model,
            safety_settings=safety_settings,
            generation_config=generation_config,
            cached_content=full_cache_name,
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
            request=self._prepare_request_gemini(
                messages=messages, stop=stop, **kwargs
            ),
            is_gemini=True,
            metadata=self.default_metadata,
            **kwargs,
        )
        return self._gemini_response_to_chat_result(response)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        if self._is_gemini_model:
            # https://cloud.google.com/vertex-ai/docs/reference/rpc/google.cloud.aiplatform.v1beta1#counttokensrequest
            _, contents = _parse_chat_history_gemini([HumanMessage(content=text)])
            response = self.prediction_client.count_tokens(
                {
                    "endpoint": self.full_model_name,
                    "model": self.full_model_name,
                    "contents": contents,
                }
            )
            return response.total_tokens
        else:
            return super().get_num_tokens(text=text)

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

    def _generate_non_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("safety_settings", None)
        params = self._prepare_params(stop=stop, stream=False, **kwargs)
        question = _get_question(messages)
        history = _parse_chat_history(messages[:-1])
        examples = kwargs.get("examples") or self.examples
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")
        if examples:
            params["examples"] = _parse_examples(examples)
        with telemetry.tool_context_manager(self._user_agent):
            chat = self._start_chat(history, **params)
            response = _completion_with_retry(
                chat.send_message,
                max_retries=self.max_retries,
                message=question.content,
                **msg_params,
            )
        usage_metadata = response.raw_prediction_response.metadata
        lc_usage = _get_usage_metadata_non_gemini(usage_metadata)
        generations = [
            ChatGeneration(
                message=AIMessage(content=candidate.text, usage_metadata=lc_usage),
                generation_info=get_generation_info(
                    candidate,
                    self._is_gemini_model,
                    usage_metadata=usage_metadata,
                ),
            )
            for candidate in response.candidates
        ]
        return ChatResult(generations=generations)

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
        if not self._is_gemini_model:
            if should_stream:
                logger.warning(
                    "ChatVertexAI does not currently support async streaming."
                )
            return await self._agenerate_non_gemini(messages, stop=stop, **kwargs)

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

    async def _agenerate_non_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("safety_settings", None)
        params = self._prepare_params(stop=stop, stream=False, **kwargs)
        question = _get_question(messages)
        history = _parse_chat_history(messages[:-1])
        examples = kwargs.get("examples") or self.examples
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")
        if examples:
            params["examples"] = _parse_examples(examples)
        with telemetry.tool_context_manager(self._user_agent):
            chat = self._start_chat(history, **params)
            response = await _acompletion_with_retry(
                chat.send_message_async,
                message=question.content,
                max_retries=self.max_retries,
                **msg_params,
            )
        usage_metadata = response.raw_prediction_response.metadata
        lc_usage = _get_usage_metadata_non_gemini(usage_metadata)
        generations = [
            ChatGeneration(
                message=AIMessage(content=candidate.text, usage_metadata=lc_usage),
                generation_info=get_generation_info(
                    candidate,
                    self._is_gemini_model,
                    usage_metadata=usage_metadata,
                ),
            )
            for candidate in response.candidates
        ]
        return ChatResult(generations=generations)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if not self._is_gemini_model:
            yield from self._stream_non_gemini(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return
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
            request=request,
            is_gemini=True,
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

    def _stream_non_gemini(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        question = _get_question(messages)
        history = _parse_chat_history(messages[:-1])
        examples = kwargs.get("examples", None)
        if examples:
            params["examples"] = _parse_examples(examples)
        with telemetry.tool_context_manager(self._user_agent):
            chat = self._start_chat(history, **params)
            responses = chat.send_message_streaming(question.content, **params)
            for response in responses:
                if run_manager:
                    run_manager.on_llm_new_token(response.text)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=response.text),
                    generation_info=get_generation_info(
                        response,
                        self._is_gemini_model,
                        usage_metadata=response.raw_prediction_response.metadata,
                    ),
                )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # TODO: Update to properly support async streaming from gemini.
        if not self._is_gemini_model:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
            return
        request = self._prepare_request_gemini(messages=messages, stop=stop, **kwargs)

        response_iter = _acompletion_with_retry(
            self.async_prediction_client.stream_generate_content,
            max_retries=self.max_retries,
            request=request,
            is_gemini=True,
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
        schema: Union[Dict, Type[BaseModel]],
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

                llm = ChatVertexAI(model_name="gemini-pro", temperature=0)
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

                llm = ChatVertexAI(model_name="gemini-pro", temperature=0)
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
                llm = ChatVertexAI(model_name="gemini-pro", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        """  # noqa: E501

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        parser: OutputParserLike

        if method == "json_mode":
            if isinstance(schema, type):
                # TODO: This gets the json schema of a pydantic model. It fails for
                # nested models because the generated schema contains $refs that the
                # gemini api doesn't support. We can implement a postprocessing function
                # that takes care of this if necessary.
                schema_json = schema.model_json_schema()
                schema_json = replace_defs_in_schema(schema_json)
                self.response_schema = schema_json
                parser = PydanticOutputParser(pydantic_object=schema)
            else:
                parser = JsonOutputParser()
                self.response_schema = schema
            self.response_mime_type = "application/json"
            llm: Runnable = self

        else:
            tool_name = _get_tool_name(schema)
            if isinstance(schema, type) and is_basemodel_subclass(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
            tool_choice = tool_name if self._is_gemini_advanced else None

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

    def _start_chat(
        self, history: _ChatHistory, **kwargs: Any
    ) -> Union[ChatSession, CodeChatSession]:
        if self.model_family == GoogleModelFamily.CODEY:
            return self.client.start_chat(
                context=history.context, message_history=history.history, **kwargs
            )
        else:
            return self.client.start_chat(message_history=history.history, **kwargs)

    def _gemini_response_to_chat_result(
        self, response: GenerationResponse
    ) -> ChatResult:
        generations = []
        usage = proto.Message.to_dict(response.usage_metadata)
        lc_usage = _get_usage_metadata_gemini(usage)
        for candidate in response.candidates:
            info = get_generation_info(candidate, is_gemini=True, usage_metadata=usage)
            message = _parse_response_candidate(candidate)
            if isinstance(message, AIMessage):
                message.usage_metadata = lc_usage
            generations.append(ChatGeneration(message=message, generation_info=info))
        if not response.candidates:
            message = AIMessage(content="")
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
                is_gemini=True,
            )
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
    if all(count == 0 for count in [input_tokens, output_tokens, total_tokens]):
        return None
    else:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def _get_usage_metadata_non_gemini(raw_metadata: dict) -> Optional[UsageMetadata]:
    """Get UsageMetadata from raw response metadata."""
    token_usage = raw_metadata.get("tokenMetadata", {})
    input_tokens = token_usage.get("inputTokenCount", {}).get("totalTokens", 0)
    output_tokens = token_usage.get("outputTokenCount", {}).get("totalTokens", 0)
    if input_tokens == 0 and output_tokens == 0:
        return None
    else:
        return UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


def _get_tool_name(tool: _ToolType) -> str:
    vertexai_tool = _format_to_gapic_tool([tool])
    return [f.name for f in vertexai_tool.function_declarations][0]
