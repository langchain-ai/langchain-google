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

import httpx
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
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_google_vertexai._anthropic_parsers import (
    ToolsOutputParser,
    _extract_tool_calls,
)
from langchain_google_vertexai._anthropic_utils import (
    _create_usage_metadata,
    _documents_in_params,
    _format_messages_anthropic,
    _make_message_chunk_from_anthropic_event,
    _thinking_in_params,
    _tools_in_params,
    convert_to_anthropic_tool,
)
from langchain_google_vertexai._base import _BaseVertexAIModelGarden, _VertexAICommon
from langchain_google_vertexai._retry import create_base_retry_decorator


def _create_retry_decorator(
    *,
    max_retries: int = 3,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
    wait_exponential_kwargs: Optional[dict[str, float]] = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Anthropic Vertex LLMs with proper tracing."""
    from anthropic import (  # type: ignore[unused-ignore, import-not-found]
        APIError,
        APITimeoutError,
        RateLimitError,
    )

    errors = [
        APIError,
        APITimeoutError,
        RateLimitError,
    ]

    return create_base_retry_decorator(
        error_types=errors,
        max_retries=max_retries,
        run_manager=run_manager,
        wait_exponential_kwargs=wait_exponential_kwargs,
    )


class VertexAIModelGarden(_BaseVertexAIModelGarden, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),
    )

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
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    max_output_tokens: int = Field(default=1024, alias="max_tokens")
    access_token: Optional[str] = None
    stream_usage: bool = True  # Whether to include usage metadata in streaming output
    credentials: Optional[Credentials] = None
    max_retries: int = Field(
        default=3, description="Number of retries for error handling."
    )
    wait_exponential_kwargs: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional dictionary with parameters for wait_exponential: "
        "- multiplier: Initial wait time multiplier (default: 1.0) "
        "- min: Minimum wait time in seconds (default: 4.0) "
        "- max: Maximum wait time in seconds (default: 10.0) "
        "- exp_base: Exponent base to use (default: 2.0) ",
    )
    timeout: Optional[Union[float, httpx.Timeout]] = Field(
        default=None,
        description="Timeout for API requests.",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Needed so that mypy doesn't flag missing aliased init args.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        from anthropic import (  # type: ignore[unused-ignore, import-not-found]
            AnthropicVertex,
            AsyncAnthropicVertex,
        )

        if self.project is None:
            raise ValueError("project is required for ChatAnthropicVertex")

        project_id: str = self.project

        # Always disable Anthropic's retries, we handle it using the retry decorator
        self.client = AnthropicVertex(
            project_id=project_id,
            region=self.location,
            base_url=self.api_endpoint,
            max_retries=0,
            access_token=self.access_token,
            credentials=self.credentials,
            timeout=self.timeout,
        )
        self.async_client = AsyncAnthropicVertex(
            project_id=project_id,
            region=self.location,
            base_url=self.api_endpoint,
            max_retries=0,
            access_token=self.access_token,
            credentials=self.credentials,
            timeout=self.timeout,
        )
        return self

    @property
    def _default_params(self):
        default_parameters = {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        return {**default_parameters, **self.model_kwargs}

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        system_message, formatted_messages = _format_messages_anthropic(
            messages, self.project
        )
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
        content = data_dict["content"]
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        if len(content) == 1 and content[0]["type"] == "text":
            msg = AIMessage(content=content[0]["text"])
        elif any(block["type"] == "tool_use" for block in content):
            tool_calls = _extract_tool_calls(content)
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
            )
        else:
            msg = AIMessage(content=content)
        # Collect token usage using the reusable function (matches langchain_anthropic)
        msg.usage_metadata = _create_usage_metadata(data.usage)
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
        """Run the LLM on the given prompt and input."""
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        retry_decorator = _create_retry_decorator(
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
        )

        @retry_decorator
        def _completion_with_retry_inner(**params: Any) -> Any:
            return self.client.messages.create(**params)

        data = _completion_with_retry_inner(**params)
        return self._format_output(data, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run the LLM on the given prompt and input."""
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        retry_decorator = _create_retry_decorator(
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
        )

        @retry_decorator
        async def _acompletion_with_retry_inner(**params: Any) -> Any:
            return await self.async_client.messages.create(**params)

        data = await _acompletion_with_retry_inner(**params)
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
        retry_decorator = _create_retry_decorator(
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
        )

        @retry_decorator
        def _stream_with_retry(**params: Any) -> Any:
            params.pop("stream", None)
            return self.client.messages.create(**params, stream=True)

        stream = _stream_with_retry(**params)
        coerce_content_to_string = (
            not _tools_in_params(params)
            and not _documents_in_params(params)
            and not _thinking_in_params(params)
        )
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
        retry_decorator = _create_retry_decorator(
            max_retries=self.max_retries,
            run_manager=run_manager,
            wait_exponential_kwargs=self.wait_exponential_kwargs,
        )

        @retry_decorator
        async def _astream_with_retry(**params: Any) -> Any:
            params.pop("stream", None)
            return await self.async_client.messages.create(**params, stream=True)

        stream = await _astream_with_retry(**params)
        coerce_content_to_string = (
            not _tools_in_params(params)
            and not _documents_in_params(params)
            and not _thinking_in_params(params)
        )
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
