from unittest.mock import MagicMock

import pytest
from anthropic import APIError

from langchain_google_vertexai.model_garden import (
    _create_retry_decorator,
)


def create_api_error():
    """Helper function to create an APIError with required arguments."""
    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.url = "test-url"
    mock_request.headers = {}
    mock_request.body = None
    return APIError(
        message="Test error",
        request=mock_request,
        body={"error": {"message": "Test error"}},
    )


def test_retry_on_errors():
    """Test that the retry decorator works with sync functions."""
    mock_llm = MagicMock()
    mock_llm.max_retries = 2
    mock_function = MagicMock(side_effect=[create_api_error(), "success"])

    decorator = _create_retry_decorator(mock_llm)
    wrapped_func = decorator(mock_function)

    result = wrapped_func()
    assert result == "success"
    assert mock_function.call_count == 2


def test_max_retries_exceeded():
    """Test that the retry decorator fails after max retries."""
    mock_llm = MagicMock()
    mock_llm.max_retries = 2
    mock_function = MagicMock(side_effect=[create_api_error(), create_api_error()])

    decorator = _create_retry_decorator(mock_llm)
    wrapped_func = decorator(mock_function)

    with pytest.raises(APIError):
        wrapped_func()
    assert mock_function.call_count == 2


@pytest.mark.asyncio
async def test_async_retry_on_errors():
    """Test that the retry decorator works with async functions."""
    mock_llm = MagicMock()
    mock_llm.max_retries = 2

    class AsyncMock:
        def __init__(self):
            self.call_count = 0

        async def __call__(self):
            self.call_count += 1
            if self.call_count == 1:
                raise create_api_error()
            return "success"

    mock_async = AsyncMock()

    decorator = _create_retry_decorator(mock_llm)
    wrapped_func = decorator(mock_async)

    result = await wrapped_func()
    assert result == "success"
    assert mock_async.call_count == 2


@pytest.mark.asyncio
async def test_async_max_retries_exceeded():
    """Test that the async retry decorator fails after max retries."""
    mock_llm = MagicMock()
    mock_llm.max_retries = 2

    class AsyncMock:
        def __init__(self):
            self.call_count = 0

        async def __call__(self):
            self.call_count += 1
            raise create_api_error()

    mock_async = AsyncMock()

    decorator = _create_retry_decorator(mock_llm)
    wrapped_func = decorator(mock_async)

    with pytest.raises(APIError):
        await wrapped_func()
    assert mock_async.call_count == 2
