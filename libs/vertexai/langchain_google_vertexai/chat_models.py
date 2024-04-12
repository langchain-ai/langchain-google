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
from langchain_core.tools import BaseTool, tool as tool_from_callable
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
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
)
from vertexai.generative_models._generative_models import (  # type: ignore
    ToolConfig,
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
    get_generation_info,
    is_codey_model,
    is_gemini_model,
)
from langchain_google_vertexai.functions_utils import (
    _format_tool_config,
    _format_tool_to_vertex_function,
    _format_tools_to_vertex_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class _ChatHistory:
    """Represents a context and a history of messages."""

    history: List[ChatMessage] = field(default_factory=list)
    context: Optional[str] = None


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
    convert_system_message_to_human_content = None
    system_instruction = None
    for i, message in enumerate(history):
        if isinstance(message, SystemMessage):
            if system_instruction is not None:
                raise ValueError(
                    "Detected more than one SystemMessage in the list of messages."
                    "Gemini APIs support the insertion of only one SystemMessage."
                )
            else:
                if convert_system_message_to_human:
                    logger.warning(
                        "gemini models released from April 2024 support"
                        "SystemMessages natively. For best performances,"
                        "when working with these models,"
                        "set convert_system_message_to_human to False"
                    )
                    convert_system_message_to_human_content = message
                    continue
                system_instruction = Content(
                    role="user", parts=_convert_to_parts(message)
                )
                continue
        elif (
            i == 0
            and isinstance(message, SystemMessage)
            and convert_system_message_to_human
        ):
            convert_system_message_to_human_content = message
            continue
        elif isinstance(message, AIMessage):
            raw_function_call = message.additional_kwargs.get("function_call")
            role = "model"
            if raw_function_call:
                function_call = FunctionCall(
                    {
                        "name": raw_function_call["name"],
                        "args": json.loads(raw_function_call["arguments"]),
                    }
                )
                gapic_part = GapicPart(function_call=function_call)
                parts = [Part._from_gapic(gapic_part)]
            else:
                parts = _convert_to_parts(message)
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message)
        elif isinstance(message, FunctionMessage):
            role = "function"
            parts = [
                Part.from_function_response(
                    name=message.name,
                    response={
                        "content": message.content,
                    },
                )
            ]
        elif isinstance(message, ToolMessage):
            role = "function"
            if (i > 0) and isinstance(history[i - 1], AIMessage):
                # message.name can be null for ToolMessage
                if history[i - 1].tool_calls:  # type: ignore
                    name = history[i - 1].tool_calls[0]["name"]  # type: ignore
                else:
                    name = message.name
            else:
                name = message.name
            parts = [
                Part.from_function_response(
                    name=name,
                    response={
                        "content": message.content,
                    },
                )
            ]
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )

        if convert_system_message_to_human_content:
            if role == "model":
                raise ValueError(
                    "SystemMessage should be followed by a HumanMessage and "
                    "not by AIMessage."
                )
            parts = _convert_to_parts(convert_system_message_to_human_content) + parts
            convert_system_message_to_human_content = None

        vertex_message = Content(role=role, parts=parts)
        vertex_messages.append(vertex_message)
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


def _get_client_with_sys_instruction(
    client: GenerativeModel,
    system_instruction: Content,
    model_name: str,
):
    if client._system_instruction != system_instruction:
        client = GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
        )
    return client


def _parse_response_candidate(
    response_candidate: "Candidate", streaming: bool = False
) -> AIMessage:
    try:
        content = response_candidate.text
    except AttributeError:
        content = ""

    additional_kwargs = {}
    first_part = response_candidate.content.parts[0]
    if first_part.function_call:
        function_call = {"name": first_part.function_call.name}
        # dump to match other function calling llm for now
        function_call_args_dict = proto.Message.to_dict(first_part.function_call)[
            "args"
        ]
        function_call["arguments"] = json.dumps(
            {k: function_call_args_dict[k] for k in function_call_args_dict}
        )
        additional_kwargs["function_call"] = function_call
        if streaming:
            tool_call_chunks = [
                ToolCallChunk(
                    name=function_call.get("name"),
                    args=function_call.get("arguments"),
                    id=function_call.get("id", str(uuid.uuid4())),
                    index=function_call.get("index"),
                )
            ]
            return AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_call_chunks,
            )
        else:
            tool_calls = []
            invalid_tool_calls = []
            try:
                tool_calls_dicts = parse_tool_calls(
                    [{"function": function_call}],
                    return_id=False,
                )
                tool_calls = [
                    ToolCall(
                        name=tool_call["name"],
                        args=tool_call["args"],
                        id=tool_call.get("id", str(uuid.uuid4())),
                    )
                    for tool_call in tool_calls_dicts
                ]
            except Exception as e:
                invalid_tool_calls = [
                    InvalidToolCall(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=function_call.get("id", str(uuid.uuid4())),
                        error=str(e),
                    )
                ]

            return AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
            )
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


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
            if float(self.model_name.split("-")[1]) > 1.0:
                return True
        except ValueError:
            pass
        except IndexError:
            pass
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
        should_stream = stream if stream is not None else self.streaming
        safety_settings = kwargs.pop("safety_settings", None)
        if should_stream:
            with telemetry.tool_context_manager(self._user_agent):
                stream_iter = self._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)

        params = self._prepare_params(stop=stop, stream=False, **kwargs)
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")

        if self._is_gemini_model:
            system_instruction, history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )
            self.client = _get_client_with_sys_instruction(
                client=self.client,
                system_instruction=system_instruction,
                model_name=self.model_name,
            )

            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            raw_tool_config = (
                params.pop("tool_config") if "tool_config" in params else None
            )
            tool_config = (
                _format_tool_config(raw_tool_config) if raw_tool_config else None
            )
            with telemetry.tool_context_manager(self._user_agent):
                response = self.client.generate_content(
                    history_gemini,
                    generation_config=params,
                    tools=tools,
                    tool_config=tool_config,
                    safety_settings=safety_settings,
                )
            generations = [
                ChatGeneration(
                    message=_parse_response_candidate(candidate),
                    generation_info=get_generation_info(
                        candidate,
                        self._is_gemini_model,
                        usage_metadata=response.to_dict().get("usage_metadata"),
                    ),
                )
                for candidate in response.candidates
            ]
        else:
            question = _get_question(messages)
            history = _parse_chat_history(messages[:-1])
            examples = kwargs.get("examples") or self.examples
            if examples:
                params["examples"] = _parse_examples(examples)
            with telemetry.tool_context_manager(self._user_agent):
                chat = self._start_chat(history, **params)
                response = chat.send_message(question.content, **msg_params)
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

        params = self._prepare_params(stop=stop, **kwargs)
        safety_settings = kwargs.pop("safety_settings", None)
        msg_params = {}
        if "candidate_count" in params:
            msg_params["candidate_count"] = params.pop("candidate_count")

        if self._is_gemini_model:
            system_instruction, history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )

            self.client = _get_client_with_sys_instruction(
                client=self.client,
                system_instruction=system_instruction,
                model_name=self.model_name,
            )
            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            raw_tool_config = (
                params.pop("tool_config") if "tool_config" in params else None
            )
            tool_config = (
                _format_tool_config(raw_tool_config) if raw_tool_config else None
            )
            with telemetry.tool_context_manager(self._user_agent):
                response = await self.client.generate_content_async(
                    history_gemini,
                    generation_config=params,
                    tools=tools,
                    tool_config=tool_config,
                    safety_settings=safety_settings,
                )
            generations = [
                ChatGeneration(
                    message=_parse_response_candidate(c),
                    generation_info=get_generation_info(
                        c,
                        self._is_gemini_model,
                        usage_metadata=response.to_dict().get("usage_metadata"),
                    ),
                )
                for c in response.candidates
            ]
        else:
            question = _get_question(messages)
            history = _parse_chat_history(messages[:-1])
            examples = kwargs.get("examples", None) or self.examples
            if examples:
                params["examples"] = _parse_examples(examples)
            with telemetry.tool_context_manager(self._user_agent):
                chat = self._start_chat(history, **params)
                response = await chat.send_message_async(question.content, **msg_params)
            generations = [
                ChatGeneration(
                    message=AIMessage(content=r.text),
                    generation_info=get_generation_info(
                        r,
                        self._is_gemini_model,
                        usage_metadata=response.raw_prediction_response.metadata,
                    ),
                )
                for r in response.candidates
            ]
        return ChatResult(generations=generations)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        if self._is_gemini_model:
            safety_settings = params.pop("safety_settings", None)
            system_instruction, history_gemini = _parse_chat_history_gemini(
                messages,
                project=self.project,
                convert_system_message_to_human=self.convert_system_message_to_human,
            )
            self.client = _get_client_with_sys_instruction(
                client=self.client,
                system_instruction=system_instruction,
                model_name=self.model_name,
            )
            # set param to `functions` until core tool/function calling implemented
            raw_tools = params.pop("functions") if "functions" in params else None
            tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
            raw_tool_config = (
                params.pop("tool_config") if "tool_config" in params else None
            )
            tool_config = (
                _format_tool_config(raw_tool_config) if raw_tool_config else None
            )
            with telemetry.tool_context_manager(self._user_agent):
                responses = self.client.generate_content(
                    history_gemini,
                    stream=True,
                    generation_config=params,
                    tools=tools,
                    tool_config=tool_config,
                    safety_settings=safety_settings,
                )
                for response in responses:
                    message = _parse_response_candidate(
                        response.candidates[0], streaming=True
                    )
                    generation_info = get_generation_info(
                        response.candidates[0],
                        self._is_gemini_model,
                        usage_metadata=response.to_dict().get("usage_metadata"),
                    )
                    if run_manager and isinstance(message.content, str):
                        run_manager.on_llm_new_token(message.content)
                    if isinstance(message, AIMessageChunk):
                        yield ChatGenerationChunk(
                            message=message,
                            generation_info=generation_info,
                        )
                    else:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=message.content,
                                additional_kwargs=message.additional_kwargs,
                            ),
                            generation_info=generation_info,
                        )
        else:
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
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        safety_settings = params.pop("safety_settings", None)
        system_instruction, history_gemini = _parse_chat_history_gemini(
            messages,
            project=self.project,
            convert_system_message_to_human=self.convert_system_message_to_human,
        )
        self.client = _get_client_with_sys_instruction(
            client=self.client,
            system_instruction=system_instruction,
            model_name=self.model_name,
        )
        raw_tools = params.pop("functions") if "functions" in params else None
        tools = _format_tools_to_vertex_tool(raw_tools) if raw_tools else None
        raw_tool_config = params.pop("tool_config") if "tool_config" in params else None
        tool_config = _format_tool_config(raw_tool_config) if raw_tool_config else None
        with telemetry.tool_context_manager(self._user_agent):
            async for chunk in await self.client.generate_content_async(
                history_gemini,
                stream=True,
                generation_config=params,
                tools=tools,
                tool_config=tool_config,
                safety_settings=safety_settings,
            ):
                message = _parse_response_candidate(chunk.candidates[0], streaming=True)
                generation_info = get_generation_info(
                    chunk.candidates[0],
                    self._is_gemini_model,
                    usage_metadata=chunk.to_dict().get("usage_metadata"),
                )
                if run_manager and isinstance(message.content, str):
                    await run_manager.on_llm_new_token(message.content)
                if isinstance(message, AIMessageChunk):
                    yield ChatGenerationChunk(
                        message=message,
                        generation_info=generation_info,
                    )
                else:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=message.content,
                            additional_kwargs=message.additional_kwargs,
                        ),
                        generation_info=generation_info,
                    )

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
            parser: OutputParserLike = PydanticOutputFunctionsParser(
                pydantic_schema=schema
            )
        else:
            parser = JsonOutputFunctionsParser()

        name = _format_tool_to_vertex_function(schema)["name"]

        if self._is_gemini_advanced:
            llm = self.bind(
                functions=[schema],
                tool_config={
                    "function_calling_config": {
                        "mode": ToolConfig.FunctionCallingConfig.Mode.ANY,
                        "allowed_function_names": [name],
                    }
                },
            )
        else:
            llm = self.bind(functions=[schema])
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
        tools: Sequence[Union[Type[BaseModel], Callable, BaseTool]],
        tool_config: Optional[Dict[str, Any]] = None,
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
        formatted_tools = []
        for schema in tools:
            if isinstance(schema, BaseTool) or (
                isinstance(schema, type) and issubclass(schema, BaseModel)
            ):
                formatted_tools.append(schema)
            elif callable(schema):
                formatted_tools.append(tool_from_callable(schema))  # type: ignore
            else:
                raise ValueError(
                    "Tool must be a BaseTool, Pydantic model, or callable."
                )
        return self.bind(functions=formatted_tools, tool_config=tool_config, **kwargs)

    def _start_chat(
        self, history: _ChatHistory, **kwargs: Any
    ) -> Union[ChatSession, CodeChatSession]:
        if not self.is_codey_model:
            return self.client.start_chat(
                context=history.context, message_history=history.history, **kwargs
            )
        else:
            return self.client.start_chat(message_history=history.history, **kwargs)
