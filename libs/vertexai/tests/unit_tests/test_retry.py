from unittest.mock import MagicMock

import pytest

from langchain_google_vertexai._retry import create_base_retry_decorator
from langchain_google_vertexai.chat_models import _completion_with_retry


class _DummyTimeoutError(Exception):
    pass


def test_create_base_retry_decorator_zero_retries_means_single_attempt() -> None:
    """Ensure max_retries=0 performs exactly one attempt and no retries."""
    calls = {"count": 0}

    def will_timeout():
        calls["count"] += 1
        raise _DummyTimeoutError("timeout")

    decorator = create_base_retry_decorator(
        error_types=[_DummyTimeoutError], max_retries=0
    )
    wrapped = decorator(will_timeout)

    with pytest.raises(_DummyTimeoutError):
        wrapped()

    assert calls["count"] == 1


def test_completion_with_retry_injects_retry_none_when_zero() -> None:
    """_completion_with_retry should pass retry=None to GAPIC when max_retries=0."""
    gen_method = MagicMock(return_value="ok")

    result = _completion_with_retry(
        gen_method,
        max_retries=0,
        request={"dummy": True},
        timeout=120,
        metadata=(),
    )

    assert result == "ok"
    # Ensure called exactly once and with retry=None passed through
    assert gen_method.call_count == 1
    kwargs = gen_method.call_args.kwargs
    assert "retry" in kwargs and kwargs["retry"] is None


def test_completion_with_retry_does_not_inject_retry_when_positive() -> None:
    """When max_retries>0, do not auto-inject retry=None."""
    gen_method = MagicMock(return_value="ok")

    result = _completion_with_retry(
        gen_method,
        max_retries=2,
        request={"dummy": True},
        timeout=30,
        metadata=(),
    )

    assert result == "ok"
    assert gen_method.call_count == 1
    kwargs = gen_method.call_args.kwargs
    assert "retry" not in kwargs
