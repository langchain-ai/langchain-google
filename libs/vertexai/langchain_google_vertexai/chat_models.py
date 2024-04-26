"""Wrapper around Google VertexAI chat-based models."""

from __future__ import annotations  # noqa

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
    TypedDict,
    overload,
)

import proto  # type: ignore[import-untyped]
from google.cloud.aiplatform_v1beta1.types.content import Part as GapicPart
from google.cloud.aiplatform_v1beta1.types.tool import FunctionCall
from google.cloud.aiplatform import telemetry

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.openai_tools import parse_tool_calls
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables import Runnable, RunnablePassthrough
from vertexai.generative_models import (  # type: ignore
    Candidate,
    Content,
    GenerativeModel,
    Part,
    Tool as VertexTool,
)
from vertexai.generative_models._generative_models import (  # type: ignore
    ToolConfig,
    SafetySettingsType,
    GenerationConfigType,
    GenerationResponse,
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

from langchain_google_vertexai._base import (
    _VertexAICommon,
)
from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
    is_codey_model,
    is_gemini_model,
)
from langchain_google_vertexai.functions_utils import (
    _format_tool_config,
    _ToolConfigDict,
    _tool_choice_to_tool_config,
    _ToolChoiceType,
    _FunctionDeclarationLike,
    _VertexToolDict,
    _format_to_vertex_tool,
    _format_functions_to_vertex_tool_dict,
)

logger = logging.getLogger(__name__)


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
    def _convert_to_prompt(part: Union[str, Dict]) -> Part:
        if isinstance(part, str):
            return Part.from_text(part)

        if not isinstance(part, Dict):
            raise ValueError(
                f"Message's content is expected to be a dict, got {type(part)}!"
            )
        if part["type"] == "text":
            return Part.from_text(part["text"])
        if part["type"] == "image_url":
            path = part["image_url"]["url"]
            return ImageBytesLoader(project=project).load_part(path)

        raise ValueError("Only text and image_url types are supported!")

    def _convert_to_parts(message: BaseMessage) -> List[Part]:
        raw_content = message.content
        if isinstance(raw_content, str):
            raw_content = [raw_content]
        return [_convert_to_prompt(part) for part in raw_content]

    vertex_messages: List[Content] = []
    system_parts: List[Part] | None = None
    system_instruction = None

    for i, message in enumerate(history):
        if isinstance(message, SystemMessage):
            if i != 0:
                raise ValueError("SystemMessage should be the first in the history.")
            if system_instruction is not None:
                raise ValueError(
                    "Detected more than one SystemMessage in the list of messages."
                    "Gemini APIs support the insertion of only one SystemMessage."
                )
            if convert_system_message_to_human:
                logger.warning(
                    "gemini models released from April 2024 support"
                    "SystemMessages natively. For best performances,"
                    "when working with these models,"
                    "set convert_system_message_to_human to False"
                )
                system_parts = _convert_to_parts(message)
                continue
            system_instruction = Content(role="user", parts=_convert_to_parts(message))
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message)
            if system_parts is not None:
                if i != 1:
                    raise ValueError(
                        "System message should be immediately followed by HumanMessage"
                    )
                parts = system_parts + parts
                system_parts = None
            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, AIMessage):
            role = "model"

            parts = []
            if message.content:
                parts = _convert_to_parts(message)

            for tc in message.tool_calls:
                function_call = FunctionCall(
                    {
                        "name": tc["name"],
                        "args": tc["args"],
                    }
                )
                gapic_part = GapicPart(function_call=function_call)
                parts.append(Part._from_gapic(gapic_part))

            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, FunctionMessage):
            role = "function"

            part = Part.from_function_response(
                name=message.name,
                response={
                    "content": message.content,
                },
            )

            prev_content = vertex_messages[-1]
            prev_content_is_function = prev_content and prev_content.role == "function"
            if prev_content_is_function:
                parts = prev_content.parts
                parts.append(part)
                # replacing last message
                vertex_messages[-1] = Content(role=role, parts=parts)
                continue

            parts = [part]

            vertex_messages.append(Content(role=role, parts=parts))
        elif isinstance(message, ToolMessage):
            role = "function"

            # message.name can be null for ToolMessage
            name = message.name
            if name is None:
                prev_message = history[i - 1] if i > 0 else None
                if isinstance(prev_message, AIMessage):
                    tool_call_id = message.tool_call_id
                    tool_call: ToolCall | None = next(
                        (t for t in prev_message.tool_calls if t["id"] == tool_call_id),
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
            part = Part.from_function_response(
                name=name,
                response={
                    "content": message.content,
                },
            )

            prev_content = vertex_messages[-1]
            prev_content_is_function = prev_content and prev_content.role == "function"
            if prev_content_is_function:
                parts = prev_content.parts
                parts.append(part)
                # replacing last message
                vertex_messages[-1] = Content(role=role, parts=parts)
                continue

            parts = [part]

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

        if text is not None:
            if content is None:
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
                tool_call_chunks.append(
                    ToolCallChunk(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=function_call.get("id", str(uuid.uuid4())),
                        index=function_call.get("index"),
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
                            ToolCall(
                                name=tool_call["name"],
                                args=tool_call["args"],
                                id=tool_call.get("id", str(uuid.uuid4())),
                            )
                            for tool_call in tool_calls_dicts
                        ]
                    )
                except Exception as e:
                    invalid_tool_calls.append(
                        InvalidToolCall(
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

    return _completion_with_retry_inner(generation_method, **kwargs)


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

    return await _completion_with_retry_inner(generation_method, **kwargs)


class ChatVertexAI(_VertexAICommon, BaseChatModel):
    """`Vertex AI` Chat large language models API."""

    model_name: str = "chat-bison"
    "Underlying model name."
    examples: Optional[List[BaseMessage]] = None
    tuned_model_name: Optional[str] = None
    """The name of a tuned model. If tuned_model_name is passed
    model_name will be used to determine the model family
    """
    convert_system_message_to_human: bool = False
    """[Deprecated] Since new Gemini models support setting a System Message,
    setting this parameter to True is discouraged.
    """

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "vertexai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        is_gemini = is_gemini_model(values["model_name"])
        safety_settings = values["safety_settings"]
        tuned_model_name = values.get("tuned_model_name")

        if safety_settings and not is_gemini:
            raise ValueError("Safety settings are only supported for Gemini models")

        cls._init_vertexai(values)

        if tuned_model_name:
            generative_model_name = values["tuned_model_name"]
        else:
            generative_model_name = values["model_name"]

        if is_gemini:
            values["client"] = GenerativeModel(
                model_name=generative_model_name,
                safety_settings=safety_settings,
            )
            values["client_preview"] = GenerativeModel(
                model_name=generative_model_name,
                safety_settings=safety_settings,
            )
        else:
            if is_codey_model(values["model_name"]):
                model_cls = CodeChatModel
                model_cls_preview = PreviewCodeChatModel
            else:
                model_cls = ChatModel
                model_cls_preview = PreviewChatModel
            values["client"] = model_cls.from_pretrained(generative_model_name)
            values["client_preview"] = model_cls_preview.from_pretrained(
                generative_model_name
            )
        return values

    @property
    def _is_gemini_advanced(self) -> bool:
        try:
            return float(self.model_name.split("-")[1]) > 1.0
        except (ValueError, IndexError):
            return False

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

        client, contents = self._gemini_client_and_contents(messages)
        params = self._gemini_params(stop=stop, **kwargs)
        with telemetry.tool_context_manager(self._user_agent):
            response = _completion_with_retry(
                client.generate_content,
                max_retries=self.max_retries,
                contents=contents,
                **params,
            )
        return self._gemini_response_to_chat_result(response)

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
        generations = [
            ChatGeneration(
                message=AIMessage(content=candidate.text),
                generation_info=get_generation_info(
                    candidate,
                    self._is_gemini_model,
                    usage_metadata=response.raw_prediction_response.metadata,
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
        if "stream" in kwargs:
            kwargs.pop("stream")
            logger.warning("ChatVertexAI does not currently support async streaming.")

        if not self._is_gemini_model:
            return await self._agenerate_non_gemini(messages, stop=stop, **kwargs)

        client, contents = self._gemini_client_and_contents(messages)
        params = self._gemini_params(stop=stop, **kwargs)
        with telemetry.tool_context_manager(self._user_agent):
            response = await _acompletion_with_retry(
                client.generate_content_async,
                max_retries=self.max_retries,
                contents=contents,
                **params,
            )
        return self._gemini_response_to_chat_result(response)

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
        generations = [
            ChatGeneration(
                message=AIMessage(content=candidate.text),
                generation_info=get_generation_info(
                    candidate,
                    self._is_gemini_model,
                    usage_metadata=response.raw_prediction_response.metadata,
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

        client, contents = self._gemini_client_and_contents(messages)
        params = self._gemini_params(stop=stop, stream=True, **kwargs)
        with telemetry.tool_context_manager(self._user_agent):
            response_iter = _completion_with_retry(
                client.generate_content,
                max_retries=self.max_retries,
                contents=contents,
                stream=True,
                **params,
            )
        for response_chunk in response_iter:
            chunk = self._gemini_chunk_to_generation_chunk(response_chunk)
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
        if not self._is_gemini_model:
            raise NotImplementedError()
        client, contents = self._gemini_client_and_contents(messages)
        params = self._gemini_params(stop=stop, stream=True, **kwargs)
        with telemetry.tool_context_manager(self._user_agent):
            async for response_chunk in await _acompletion_with_retry(
                client.generate_content_async,
                max_retries=self.max_retries,
                contents=contents,
                stream=True,
                **params,
            ):
                chunk = self._gemini_chunk_to_generation_chunk(response_chunk)
                if run_manager and isinstance(chunk.message.content, str):
                    await run_manager.on_llm_new_token(chunk.message.content)
                yield chunk

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

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

        Returns:
            A Runnable that takes any ChatModel input. If include_raw is True then a
            dict with keys — raw: BaseMessage, parsed: Optional[_DictOrPydantic],
            parsing_error: Optional[BaseException]. If include_raw is False then just
            _DictOrPydantic is returned, where _DictOrPydantic depends on the schema.
            If schema is a Pydantic class then _DictOrPydantic is the Pydantic class.
            If schema is a dict then _DictOrPydantic is a dict.

        Example: Pydantic schema, exclude raw:
            .. code-block:: python

                from langchain_core.pydantic_v1 import BaseModel
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

                from langchain_core.pydantic_v1 import BaseModel
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

                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool
                from langchain_google_vertexai import ChatVertexAI

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
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
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            parser = JsonOutputToolsParser()
        llm = self.bind_tools([schema], tool_choice=self._is_gemini_advanced)
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
        tools: Sequence[Union[_FunctionDeclarationLike, VertexTool]],
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
        vertexai_tools: List[_VertexToolDict] = []
        vertexai_functions = []
        for schema in tools:
            if isinstance(schema, VertexTool):
                vertexai_tools.append(
                    {"function_declarations": schema.to_dict()["function_declarations"]}
                )
            elif isinstance(schema, dict) and "function_declarations" in schema:
                vertexai_tools.append(cast(_VertexToolDict, schema))
            else:
                vertexai_functions.append(schema)
        vertexai_tools.append(_format_functions_to_vertex_tool_dict(vertexai_functions))
        if tool_choice:
            all_names = [
                f["name"] for vt in vertexai_tools for f in vt["function_declarations"]
            ]
            tool_config = _tool_choice_to_tool_config(tool_choice, all_names)
        # Bind dicts for easier serialization/deserialization.
        return self.bind(tools=vertexai_tools, tool_config=tool_config, **kwargs)

    def _start_chat(
        self, history: _ChatHistory, **kwargs: Any
    ) -> Union[ChatSession, CodeChatSession]:
        if not self.is_codey_model:
            return self.client.start_chat(
                context=history.context, message_history=history.history, **kwargs
            )
        else:
            return self.client.start_chat(message_history=history.history, **kwargs)

    def _gemini_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Union[_VertexToolDict, VertexTool]]] = None,
        functions: Optional[List[_FunctionDeclarationLike]] = None,
        tool_config: Optional[Union[_ToolConfigDict, ToolConfig]] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        **kwargs: Any,
    ) -> _GeminiGenerateContentKwargs:
        generation_config = self._prepare_params(stop=stop, stream=stream, **kwargs)
        if tools:
            tools = [_format_to_vertex_tool(tool) for tool in tools]
        elif functions:
            tools = [_format_to_vertex_tool(functions)]
        else:
            pass

        if tool_config and not isinstance(tool_config, ToolConfig):
            tool_config = _format_tool_config(cast(_ToolConfigDict, tool_config))

        return _GeminiGenerateContentKwargs(
            generation_config=generation_config,
            tools=tools,
            tool_config=tool_config,
            safety_settings=safety_settings,
        )

    def _gemini_client_and_contents(
        self, messages: List[BaseMessage]
    ) -> tuple[GenerativeModel, list[Content]]:
        system, contents = _parse_chat_history_gemini(
            messages,
            project=self.project,
            convert_system_message_to_human=self.convert_system_message_to_human,
        )
        # TODO: Store default client params explicitly so private params don't have to
        # be accessed, like _safety_settings.
        client = GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            safety_settings=self.client._safety_settings,
        )
        return client, contents

    def _gemini_response_to_chat_result(
        self, response: GenerationResponse
    ) -> ChatResult:
        generations = []
        usage = response.to_dict().get("usage_metadata")
        for candidate in response.candidates:
            info = get_generation_info(candidate, is_gemini=True, usage_metadata=usage)
            message = _parse_response_candidate(candidate)
            generations.append(ChatGeneration(message=message, generation_info=info))
        if not response.candidates:
            message = AIMessage(content="")
            if usage:
                generation_info = {"usage_metadata": usage}
            else:
                generation_info = {}
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        return ChatResult(generations=generations)

    def _gemini_chunk_to_generation_chunk(
        self, response_chunk: GenerationResponse
    ) -> ChatGenerationChunk:
        # return an empty completion message if there's no candidates
        usage_metadata = response_chunk.to_dict().get("usage_metadata")
        if not response_chunk.candidates:
            message = AIMessageChunk(content="")
            if usage_metadata:
                generation_info = {"usage_metadata": usage_metadata}
            else:
                generation_info = {}
        else:
            top_candidate = response_chunk.candidates[0]
            message = _parse_response_candidate(top_candidate, streaming=True)
            generation_info = get_generation_info(
                top_candidate,
                is_gemini=True,
                usage_metadata=usage_metadata,
            )
        return ChatGenerationChunk(
            message=message,
            generation_info=generation_info,
        )
