from __future__ import annotations

import asyncio
import json
import uuid
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
    overload,
)

import requests
from google import auth
from google.auth.credentials import Credentials
from google.auth.transport import requests as auth_requests
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
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk
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
from langchain_core.utils.function_calling import convert_to_openai_function

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
from langchain_google_vertexai._base import (
    _BaseVertexAIModelGarden,
    _VertexAIBase,
    _VertexAICommon,
)
from langchain_google_vertexai._utils import VertexMaaSModelFamily


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


@overload
def _parse_response_candidate(
    response_candidate: Dict[str, str], streaming: Literal[False] = False
) -> AIMessage:
    ...


@overload
def _parse_response_candidate(
    response_candidate: Dict[str, str], streaming: Literal[True]
) -> AIMessageChunk:
    ...


def _parse_response_candidate(
    response_candidate: Dict[str, str], streaming: bool = False
) -> AIMessage:
    content = response_candidate["content"]
    role = response_candidate["role"]
    if role != "assistant":
        raise ValueError(f"Role in response is {role}, expected 'assistant'!")
    tool_calls = []
    tool_call_chunks = []

    response_json = None
    try:
        response_json = json.loads(response_candidate["content"])
    except ValueError:
        pass
    if response_json and "name" in response_json:
        function_name = response_json["name"]
        function_args = response_json.get("parameters", None)
        if streaming:
            tool_call_chunks.append(
                tool_call_chunk(
                    name=function_name, args=function_args, id=str(uuid.uuid4())
                )
            )
        else:
            tool_calls.append(
                create_tool_call(
                    name=function_name, args=function_args, id=str(uuid.uuid4())
                )
            )
        content = ""

    if streaming:
        return AIMessageChunk(
            content=content,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
    )


class VertexMaaS(_VertexAIBase, BaseChatModel):
    """Google Cloud Vertex AI Model-as-a-Service chat model integration.

    For more information, see:
        https://cloud.google.com/blog/products/ai-machine-learning/llama-3-1-on-vertex-ai
        and https://cloud.google.com/blog/products/ai-machine-learning/codestral-and-mistral-large-v2-on-vertex-ai


    Setup:
        You need to enable a corresponding MaaS model (Google Cloud UI console ->
        Vertex AI -> Model Garden -> search for a model you need and click enable)

        You must have the langchain-google-vertexai Python package installed
        .. code-block:: bash

            pip install -U langchain-google-vertexai

        And either:
            - Have credentials configured for your environment
                (gcloud, workload identity, etc...)
            - Store the path to a service account JSON file as the
                GOOGLE_APPLICATION_CREDENTIALS environment variable

        This codebase uses the google.auth library which first looks for the application
        credentials variable mentioned above, and then looks for system-level auth.

        For more information, see:
        https://cloud.google.com/docs/authentication/application-default-credentials#GAC
        and https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth.

    Key init args — completion params:
        model: str
            Name of VertexMaaS model to use. Currently three models are supported:
            "meta/llama3-405b-instruct-maas", "mistral-nemo@2407" and
            "mistral-large@2407"
        append_tools_to_system_message: bool
            Whether to append tools to a system message (useful for Llama 3.1 tool
            calling only)


    Key init args — client params:
        credentials: Optional[google.auth.credentials.Credentials]
            The default custom credentials to use when making API calls. If not
            provided, credentials will be ascertained from the environment.
        project: Optional[str]
            The default GCP project to use when making Vertex API calls.
        location: str = "us-central1"
            The default location to use when making API calls.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_google_vertexai import VertexMaaS

            llm = VertexMaaS(
                model="gemini-1.5-flash-001",
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

            AIMessage(content="J'adore programmer. \n", id='run-925ce305-2268-44c4-875f-dde9128520ad-0')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='J', id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content="'adore programmer. \n", id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')
            AIMessageChunk(content='', id='run-9df01d73-84d9-42db-9d6b-b1466a019e89')

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content="J'adore programmer. \n", id='run-b7f7492c-4cb5-42d0-8fc3-dce9b293b0fb')

    """  # noqa: E501

    append_tools_to_system_message: bool = False
    "Whether to append tools to the system message or not."
    model_family: Optional[VertexMaaSModelFamily] = None

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        family = VertexMaaSModelFamily(values["model_name"])
        values["model_family"] = family
        if family == VertexMaaSModelFamily.MISTRAL:
            model = values["model_name"].split("@")[0]
            values["full_model_name"] = values["model_name"]
            values["model_name"] = model
        return values

    @property
    def token(self) -> str:
        if self.credentials:
            if not self.credentials.token:
                request = auth_requests.Request()
                self.credentials.refresh(request)
            return self.credentials.token
        credentials, _ = auth.default()
        if not credentials.token:
            request = auth_requests.Request()
            credentials.refresh(request)
        return credentials.token

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "x-goog-api-client": self._libary_version,
            "user_agent": self._user_agent,
        }

    def get_url(self, stream: bool = False) -> str:
        if self.model_family == VertexMaaSModelFamily.LLAMA:
            url_part = "endpoints/openapi/chat/completions"
            version = "v1beta1"
        else:
            version = "v1"
            if stream:
                url_part = (
                    f"publishers/mistralai/models/{self.model_name}:streamRawPredict"
                )
            else:
                url_part = f"publishers/mistralai/models/{self.model_name}:rawPredict"
        return (
            f"https://{self.location}-aiplatform.googleapis.com/{version}/projects/"
            f"{self.project}/locations/{self.location}/{url_part}"
        )

    def _convert_messages(
        self, messages: List[BaseMessage], tools: Optional[List[BaseTool]] = None
    ) -> List[Dict[str, Any]]:
        converted_messages: List[Dict[str, Any]] = []
        if tools and not self.append_tools_to_system_message:
            raise ValueError(
                "If providing tools, either format system message yourself or "
                "append_tools_to_system_message to True!"
            )
        elif tools:
            tools_str = "\n".join(
                [json.dumps(convert_to_openai_function(t)) for t in tools]
            )
            formatted_system_message = (
                "You are an assistant with access to the following tools:\n\n"
                f"{tools_str}\n\n"
                "If you decide to use a tool, please respond with a JSON for a "
                "function call with its proper arguments that best answers the "
                "given prompt.\nRespond in the format "
                '{"name": function name, "parameters": dictionary '
                "of argument name and its value}. Do not use variables.\n"
                "Do not provide any additional comments when calling a tool.\n"
                "Do not mention tools to the user when preparing the final answer."
            )
            message = messages[0]
            if not isinstance(message, SystemMessage):
                converted_messages.append(
                    {"role": "system", "content": formatted_system_message}
                )
            else:
                converted_messages.append(
                    {
                        "role": "system",
                        "content": str(message.content)
                        + "\n"
                        + formatted_system_message,
                    }
                )

        for i, message in enumerate(messages):
            if tools and isinstance(message, SystemMessage) and i == 0:
                continue
            if isinstance(message, AIMessage):
                converted_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            elif isinstance(message, HumanMessage):
                converted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, SystemMessage):
                converted_messages.append(
                    {"role": "system", "content": message.content}
                )
            elif isinstance(message, ToolMessage):
                # we also need to format a previous message if we got a tool result
                prev_message = messages[i - 1]
                if not isinstance(prev_message, AIMessage):
                    raise ValueError("ToolMessage should follow AIMessage only!")
                _ = converted_messages[-1].pop("content", None)
                tool_calls = []
                for tool_call in prev_message.tool_calls:
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": tool_call["id"],
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call.get("args", {})),
                            },
                        }
                    )
                converted_messages[-1]["tool_calls"] = tool_calls
                if len(tool_calls) > 1:
                    raise ValueError(
                        "Only a single function call per turn is supported!"
                    )
                converted_messages.append(
                    {
                        "role": "tool",
                        "name": message.name,
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            else:
                raise ValueError(f"Message type {type(message)} is not yet supported!")
        return converted_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        *,
        tools: Optional[List[BaseTool]] = None,
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
        formatted_messages = messages
        if self.model_family != VertexMaaSModelFamily.LLAMA and tools:
            raise ValueError("Tools are supported only for Llama 3.1!")

        if stream is True:
            stream_iter = self._stream(
                formatted_messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                **kwargs,
            )
            return generate_from_stream(stream_iter)

        converted_messages = self._convert_messages(formatted_messages, tools=tools)

        data = {
            "model": self.model_name,
            "stream": False,
            "messages": converted_messages,
        }
        response = requests.post(self.get_url(), headers=self.headers, json=data)
        result = response.json()
        if response.status_code != 200:
            raise ValueError(json.dumps(result))

        generations = []
        for candidate in result["choices"]:
            message = _parse_response_candidate(candidate["message"])
            generations.append(ChatGeneration(message=message))

        return ChatResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "vertexai_model_garden_maas"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        converted_messages = self._convert_messages(messages, tools=tools)
        data = {
            "model": self.model_name,
            "stream": True,
            "messages": converted_messages,
        }
        response = requests.post(
            self.get_url(stream=True), headers=self.headers, json=data, stream=True
        )
        buffer = ""

        def try_parse_chunk(buffer):
            try:
                if buffer.startswith("data: "):
                    json_obj, index = json.JSONDecoder().raw_decode(buffer[6:])
                    index += 6
                else:
                    json_obj, index = json.JSONDecoder().raw_decode(buffer)
                chunk = _parse_response_candidate(
                    json_obj["choices"][0]["delta"], streaming=True
                )
                if run_manager and isinstance(chunk.content, str):
                    run_manager.on_llm_new_token(chunk.content)
                return chunk, index
            except json.JSONDecodeError:
                pass
            return None, None

        for raw_chunk in response.iter_content(decode_unicode=True):
            buffer += raw_chunk
            chunk, index = try_parse_chunk(buffer)
            if index:
                yield ChatGenerationChunk(
                    message=chunk,
                    generation_info={},
                )
                buffer = buffer[index:]
        chunk, _ = try_parse_chunk(buffer)
        if chunk:
            yield ChatGenerationChunk(
                message=chunk,
                generation_info={},
            )
        return
