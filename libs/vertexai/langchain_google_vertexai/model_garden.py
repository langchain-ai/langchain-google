from __future__ import annotations

import asyncio
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from google.auth.credentials import Credentials
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool

from langchain_google_vertexai._anthropic_parsers import (
    ToolsOutputParser,
    _extract_tool_calls,
)
from langchain_google_vertexai._anthropic_utils import (
    _format_messages_anthropic,
    _make_message_chunk_from_anthropic_event,
    _tools_in_params,
    convert_to_anthropic_tool,
)
from langchain_google_vertexai._base import _BaseVertexAIModelGarden, _VertexAICommon


class VertexAIModelGarden(_BaseVertexAIModelGarden, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    # Needed so that mypy doesn't flag missing aliased init args.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)

        if self.single_example_per_request and len(instances) > 1:
            results = []
            for instance in instances:
                response = self.client.predict(
                    endpoint=self.endpoint_path, instances=[instance]
                )
                results.append(self._parse_prediction(response.predictions[0]))
            return LLMResult(
                generations=[[Generation(text=result)] for result in results]
            )

        response = self.client.predict(endpoint=self.endpoint_path, instances=instances)
        return self._parse_response(response)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        if self.single_example_per_request and len(instances) > 1:
            responses = []
            for instance in instances:
                responses.append(
                    self.async_client.predict(
                        endpoint=self.endpoint_path, instances=[instance]
                    )
                )

            responses = await asyncio.gather(*responses)
            return LLMResult(
                generations=[
                    [Generation(text=self._parse_prediction(response.predictions[0]))]
                    for response in responses
                ]
            )

        response = await self.async_client.predict(
            endpoint=self.endpoint_path, instances=instances
        )
        return self._parse_response(response)


class ChatAnthropicVertex(_VertexAICommon, BaseChatModel):
    async_client: Any = None  #: :meta private:
    model_name: Optional[str] = Field(default=None, alias="model")  # type: ignore[assignment]
    "Underlying model name."
    max_output_tokens: int = Field(default=1024, alias="max_tokens")
    access_token: Optional[str] = None
    stream_usage: bool = True  # Whether to include usage metadata in streaming output
    credentials: Optional[Credentials] = None

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    # Needed so that mypy doesn't flag missing aliased init args.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        from anthropic import (  # type: ignore
            AnthropicVertex,
            AsyncAnthropicVertex,
        )

        values["client"] = AnthropicVertex(
            project_id=values["project"],
            region=values["location"],
            max_retries=values["max_retries"],
            access_token=values["access_token"],
            credentials=values["credentials"],
        )
        values["async_client"] = AsyncAnthropicVertex(
            project_id=values["project"],
            region=values["location"],
            max_retries=values["max_retries"],
            access_token=values["access_token"],
            credentials=values["credentials"],
        )
        return values

    @property
    def _default_params(self):
        return {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        system_message, formatted_messages = _format_messages_anthropic(messages)
        params = self._default_params
        params.update(kwargs)
        if kwargs.get("model_name"):
            params["model"] = params["model_name"]
        if kwargs.get("model"):
            params["model"] = kwargs["model"]
        params.pop("model_name", None)
        params.update(
            {
                "system": system_message,
                "messages": formatted_messages,
                "stop_sequences": stop,
            }
        )
        return {k: v for k, v in params.items() if v is not None}

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        data_dict = data.model_dump()
        content = [c for c in data_dict["content"] if c["type"] != "tool_use"]
        content = content[0]["text"] if len(content) == 1 else content
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        tool_calls = _extract_tool_calls(data_dict["content"])
        if tool_calls:
            msg = AIMessage(content=content, tool_calls=tool_calls)
        else:
            msg = AIMessage(content=content)
        # Collect token usage
        msg.usage_metadata = {
            "input_tokens": data.usage.input_tokens,
            "output_tokens": data.usage.output_tokens,
            "total_tokens": data.usage.input_tokens + data.usage.output_tokens,
        }
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output=llm_output,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        data = self.client.messages.create(**params)
        return self._format_output(data, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        data = await self.async_client.messages.create(**params)
        return self._format_output(data, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat-vertexai"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        stream = self.client.messages.create(**params, stream=True)
        coerce_content_to_string = not _tools_in_params(params)
        for event in stream:
            msg = _make_message_chunk_from_anthropic_event(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
            )
            if msg is not None:
                chunk = ChatGenerationChunk(message=msg)
                if run_manager and isinstance(msg.content, str):
                    run_manager.on_llm_new_token(msg.content, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        stream = await self.async_client.messages.create(**params, stream=True)
        coerce_content_to_string = not _tools_in_params(params)
        async for event in stream:
            msg = _make_message_chunk_from_anthropic_event(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
            )
            if msg is not None:
                chunk = ChatGenerationChunk(message=msg)
                if run_manager and isinstance(msg.content, str):
                    await run_manager.on_llm_new_token(msg.content, chunk=chunk)
                yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[Dict[str, str], Literal["any", "auto"], str]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model"""

        formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]
        if not tool_choice:
            pass
        elif isinstance(tool_choice, dict):
            kwargs["tool_choice"] = tool_choice
        elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
            kwargs["tool_choice"] = {"type": tool_choice}
        elif isinstance(tool_choice, str):
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        else:
            raise ValueError(
                f"Unrecognized 'tool_choice' type {tool_choice=}. Expected dict, "
                f"str, or None."
            )
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema."""

        tool_name = convert_to_anthropic_tool(schema)["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            output_parser = ToolsOutputParser(
                first_tool_only=True, pydantic_schemas=[schema]
            )
        else:
            output_parser = ToolsOutputParser(first_tool_only=True, args_only=True)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
