import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional, Union

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from vertexai.preview import caching  # type: ignore

from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _parse_chat_history_gemini,
)
from langchain_google_vertexai.functions_utils import (
    _format_to_gapic_tool,
    _format_tool_config,
    _ToolConfigDict,
    _ToolsType,
)


def create_context_cache(
    model: ChatVertexAI,
    messages: List[BaseMessage],
    expire_time: Optional[datetime] = None,
    time_to_live: Optional[timedelta] = None,
    tools: Optional[_ToolsType] = None,
    tool_config: Optional[_ToolConfigDict] = None,
) -> str:
    """Creates a cache for content in some model.

    Args:
        model: ChatVertexAI model. Must be at least gemini-1.5 pro or flash.
        messages: List of messages to cache.
        expire_time:  Timestamp of when this resource is considered expired.
        At most one of expire_time and ttl can be set. If neither is set, default TTL
            on the API side will be used (currently 1 hour).
        time_to_live:  The TTL for this resource. If provided, the expiration time is
        computed: created_time + TTL.
        At most one of expire_time and ttl can be set. If neither is set, default TTL
            on the API side will be used (currently 1 hour).
        tools:  A list of tool definitions to bind to this chat model.
            Can be a pydantic model, callable, or BaseTool. Pydantic
            models, callables, and BaseTools will be automatically converted to
            their schema dictionary representation.
        tool_config: Optional. Immutable. Tool config. This config is shared for all
            tools.

    Raises:
        ValueError: If model doesn't support context catching.

    Returns:
        String with the identificator of the created cache.
    """

    if not model._is_gemini_advanced:
        error_msg = f"Model {model.full_model_name} doesn't support context catching"
        raise ValueError(error_msg)

    system_instruction, contents = _parse_chat_history_gemini(
        messages, ImageBytesLoader(project=model.project)
    )

    if tool_config:
        tool_config = _format_tool_config(tool_config)

    if tools is not None:
        tools = [_format_to_gapic_tool(tools)]

    cached_content = caching.CachedContent.create(
        model_name=model.full_model_name,
        system_instruction=system_instruction,
        contents=contents,
        ttl=time_to_live,
        expire_time=expire_time,
        tool_config=tool_config,
        tools=tools,
    )

    return cached_content.name


def create_base_retry_decorator(
    error_types: list[type[BaseException]],
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator for a given LLM and provided a list of error types.

    Args:
        error_types: List of error types to retry on.
        max_retries: Number of retries. Default is 1.
        run_manager: Callback manager for the run. Default is None.

    Returns:
        A retry decorator.
    """
    logger = logging.getLogger(__name__)
    _logging = before_sleep_log(logger, logging.WARNING)

    def _before_sleep(retry_state: RetryCallState) -> None:
        _logging(retry_state)
        if run_manager:
            retry_d: dict[str, Any] = {
                "slept": retry_state.idle_for,
                "attempt": retry_state.attempt_number,
            }
            if retry_state.outcome is None:
                retry_d["outcome"] = "N/A"
            elif retry_state.outcome.failed:
                retry_d["outcome"] = "failed"
                exception = retry_state.outcome.exception()
                retry_d["exception"] = str(exception)
                retry_d["exception_type"] = exception.__class__.__name__
            else:
                retry_d["outcome"] = "success"
                retry_d["result"] = str(retry_state.outcome.result())
            if isinstance(run_manager, AsyncCallbackManagerForLLMRun):
                coro = run_manager.on_retry(retry_state)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(coro)
                    else:
                        asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Error in on_retry: {e}")
            else:
                run_manager.metadata.update({"retry_state": retry_d})
                run_manager.on_retry(retry_state)

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    retry_instance: retry_base = retry_if_exception_type(error_types[0])
    for error in error_types[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_instance,
        before_sleep=_before_sleep,
    )
