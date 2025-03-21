import asyncio
import logging
from typing import Any, Callable, Optional, Union

from google.api_core.exceptions import (
    GoogleAPICallError,
    InvalidArgument,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_base,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def create_base_retry_decorator(
    error_types: list[type[BaseException]],
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
    wait_exponential_kwargs: Optional[dict[str, float]] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator for a given LLM and provided a list of error types.

    Args:
        error_types: List of error types to retry on.
        max_retries: Number of retries. Default is 1.
        run_manager: Callback manager for the run. Default is None.
        wait_exponential_kwargs: Optional dictionary with parameters:
            - multiplier: Initial wait time multiplier (default: 1.0)
            - min: Minimum wait time in seconds (default: 4.0)
            - max: Maximum wait time in seconds (default: 10.0)
            - exp_base: Exponent base to use (default: 2.0)

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

    # Default wait parameters
    wait_params = {
        "multiplier": 1.0,
        "min": 4.0,
        "max": 10.0,
        "exp_base": 2.0,
    }

    # Update with user-provided parameters
    if wait_exponential_kwargs:
        wait_params.update(wait_exponential_kwargs)

    def get_google_api_call_error_retry_instance():
        # Not retrying for InvalidArgument.
        # Retry for other error types having base class as GoogleAPICallError.
        return retry_if_exception_type(
            GoogleAPICallError
        ) & retry_if_not_exception_type(InvalidArgument)

    retry_instance: retry_base

    for index, error in enumerate(error_types):
        if index == 0:
            if error is GoogleAPICallError:
                retry_instance = get_google_api_call_error_retry_instance()
            else:
                retry_instance = retry_if_exception_type(error)
        else:
            if error is GoogleAPICallError:
                retry_instance = (retry_instance) | (
                    get_google_api_call_error_retry_instance()
                )
            else:
                retry_instance = (retry_instance) | (retry_if_exception_type(error))

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(**wait_params),
        retry=retry_instance,
        before_sleep=_before_sleep,
    )
