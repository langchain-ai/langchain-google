from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import (
    Any,
    Literal,
    cast,
    overload,
)

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
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)

from langchain_google_vertexai.model_garden_maas._base import (
    _BaseVertexMaasModelGarden,
    acompletion_with_retry,
    completion_with_retry,
)


@overload
def _parse_response_candidate_llama(
    response_candidate: dict[str, str], streaming: Literal[False] = False
) -> AIMessage: ...


@overload
def _parse_response_candidate_llama(
    response_candidate: dict[str, str], streaming: Literal[True]
) -> AIMessageChunk: ...


def _parse_response_candidate_llama(
    response_candidate: dict[str, Any], streaming: bool = False
) -> AIMessage:
    content = response_candidate.get("content", "")
    role = response_candidate["role"]
    if role != "assistant":
        msg = f"Role in response is {role}, expected 'assistant'!"
        raise ValueError(msg)
    tool_calls = []
    tool_call_chunks = []

    response_json = None
    try:
        if "content" in response_candidate:
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
    elif "tool_calls" in response_candidate:
        for tool_call in response_candidate["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"].get("arguments", None)
            if function_args is not None:
                with contextlib.suppress(ValueError):
                    function_args = json.loads(function_args)
            if streaming:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=function_name,
                        args=function_args,
                        id=str(uuid.uuid4()),
                    )
                )
            else:
                tool_calls.append(
                    create_tool_call(
                        name=tool_call["function"]["name"],
                        args=function_args,
                        id=str(uuid.uuid4()),
                    )
                )

    if streaming:
        return AIMessageChunk(
            content=content,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
    )


class VertexModelGardenLlama(_BaseVertexMaasModelGarden, BaseChatModel):
    r"""Integration for Llama 3.1 on Google Cloud Vertex AI Model-as-a-Service.

    [More information](https://cloud.google.com/blog/products/ai-machine-learning/llama-3-1-on-vertex-ai)

    Setup:
        You need to enable a corresponding MaaS model (Google Cloud UI console ->
        Vertex AI -> Model Garden -> search for a model you need and click enable)

        And either:
            - Have credentials configured for your environment (gcloud, workload
                identity, etc...)
            - Store the path to a service account JSON file as the
                `GOOGLE_APPLICATION_CREDENTIALS` environment variable

        This codebase uses the `google.auth` library which first looks for the
        application credentials variable mentioned above, and then looks for system-level auth.

    Key init args — completion params:
        model: str
            Name of VertexMaaS model to use (`'meta/llama3-405b-instruct-maas'`)
        append_tools_to_system_message: bool
            Whether to append tools to a system message

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
        ```python
        from langchain_google_vertexai import VertexMaaS

        llm = VertexModelGardenLlama(
            model="meta/llama3-405b-instruct-maas",
            # other params...
        )
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
            content="J'adore programmer. \n",
            id="run-925ce305-2268-44c4-875f-dde9128520ad-0",
        )
        ```

    Stream:
        ```python
        for chunk in llm.stream(messages):
            print(chunk)
        ```

        ```python
        AIMessageChunk(content="J", id="run-9df01d73-84d9-42db-9d6b-b1466a019e89")
        AIMessageChunk(
            content="'adore programmer. \n",
            id="run-9df01d73-84d9-42db-9d6b-b1466a019e89",
        )
        AIMessageChunk(content="", id="run-9df01d73-84d9-42db-9d6b-b1466a019e89")
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
            content="J'adore programmer. \n",
            id="run-b7f7492c-4cb5-42d0-8fc3-dce9b293b0fb",
        )
        ```
    """  # noqa: E501

    def _convert_messages(
        self, messages: list[BaseMessage], tools: list[BaseTool] | None = None
    ) -> list[dict[str, Any]]:
        converted_messages: list[dict[str, Any]] = []
        if tools and not self.append_tools_to_system_message:
            msg = (
                "If providing tools, either format system message yourself or "
                "append_tools_to_system_message to True!"
            )
            raise ValueError(msg)
        if tools:
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
                    msg = "ToolMessage should follow AIMessage only!"
                    raise ValueError(msg)
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
                    msg = "Only a single function call per turn is supported!"
                    raise ValueError(msg)
                converted_messages.append(
                    {
                        "role": "tool",
                        "name": message.name,
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            else:
                msg = f"Message type {type(message)} is not yet supported!"
                raise ValueError(msg)
        return converted_messages

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        *,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages. Code chat
                does not support context.
            stop: List of stop words.
            run_manager: The `CallbackManager` for LLM run. Not used at the moment.
            stream: Whether to use the streaming endpoint.

        Returns:
            `ChatResult` that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        if stream is True:
            return generate_from_stream(
                self._stream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    tools=tools,
                    **kwargs,
                )
            )

        converted_messages = self._convert_messages(messages, tools=tools)

        response = completion_with_retry(self, messages=converted_messages, **kwargs)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        *,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stream:
            stream_iter = self._astream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        converted_messages = self._convert_messages(messages, tools=tools)
        response = await acompletion_with_retry(
            self, messages=converted_messages, run_manager=run_manager, **kwargs
        )
        return self._create_chat_result(response)

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        token_usage = response.get("usage", {})
        for candidate in response["choices"]:
            finish_reason = response.get("finish_reason")
            message = _parse_response_candidate_llama(candidate["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            gen = ChatGeneration(
                message=message,
                generation_info={"finish_reason": finish_reason},
            )
            generations.append(gen)

        llm_output = {"token_usage": token_usage, "model": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "vertexai_model_garden_maas_llama"

    def _parse_chunk(self, chunk: dict) -> AIMessageChunk:
        chunk_delta = chunk["choices"][0]["delta"]
        content = chunk_delta.get("content", "")
        if chunk_delta.get("role") != "assistant":
            msg = f"Got chunk with non-assistant role: {chunk_delta}"
            raise ValueError(msg)
        additional_kwargs = {}
        if raw_tool_calls := chunk_delta.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            try:
                tool_call_chunks = []
                for raw_tool_call in raw_tool_calls:
                    if not raw_tool_call.get("index") and not raw_tool_call.get("id"):
                        tool_call_id = str(uuid.uuid4())
                    else:
                        tool_call_id = raw_tool_call.get("id")
                    tool_call_chunks.append(
                        tool_call_chunk(
                            name=raw_tool_call["function"].get("name"),
                            args=raw_tool_call["function"].get("arguments"),
                            id=tool_call_id,
                            index=raw_tool_call.get("index"),
                        )
                    )
            except KeyError:
                pass
        else:
            tool_call_chunks = []
        if token_usage := chunk.get("usage"):
            usage_metadata = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        converted_messages = self._convert_messages(messages, tools=tools)
        params = {**kwargs, "stream": True, "headers_content_type": "text/event-stream"}

        for chunk in completion_with_retry(
            self, messages=converted_messages, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            message = self._parse_chunk(chunk)
            gen_chunk = ChatGenerationChunk(message=message)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=cast("str", message.content), chunk=gen_chunk
                )
            yield gen_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        converted_messages = self._convert_messages(messages, tools=tools)
        params = {**kwargs, "stream": True, "headers_content_type": "text/event-stream"}

        async for chunk in await acompletion_with_retry(
            self, messages=converted_messages, run_manager=run_manager, **params
        ):
            if len(chunk["choices"]) == 0:
                continue
            message = self._parse_chunk(chunk)
            gen_chunk = ChatGenerationChunk(message=message)
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=cast("str", message.content), chunk=gen_chunk
                )
            yield gen_chunk

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_function(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
