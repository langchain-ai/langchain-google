"""Tests configuration to be executed before tests execution."""

from collections.abc import Generator
from typing import Any
import os

import pytest

_RELEASE_FLAG = "release"
_GPU_FLAG = "gpu"
_LONG_FLAG = "long"
_EXTENDED_FLAG = "extended"

_PYTEST_FLAGS = [_RELEASE_FLAG, _GPU_FLAG, _LONG_FLAG, _EXTENDED_FLAG]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add flags accepted by pytest CLI.

    Args:
        parser: The pytest parser object.
    """
    for flag in _PYTEST_FLAGS:
        parser.addoption(
            f"--{flag}", action="store_true", default=False, help=f"run {flag} tests"
        )


def pytest_configure(config: pytest.Config) -> None:
    """Add pytest custom configuration.

    Args:
        config: The pytest config object.
    """
    for flag in _PYTEST_FLAGS:
        config.addinivalue_line(
            "markers", f"{flag}: mark test to run as {flag} only test"
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests with a marker from our list that were not explicitly invoked.

    Args:
        config: The pytest config object.
        items: The list of tests to be executed.
    """
    for item in items:
        keywords = list(set(item.keywords).intersection(_PYTEST_FLAGS))
        if keywords and not any(config.getoption(f"--{kw}") for kw in keywords):
            skip = pytest.mark.skip(reason=f"need --{keywords[0]} option to run")
            item.add_marker(skip)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize backend based on environment variable.

    Args:
        metafunc: The pytest metafunc object.
    """
    if "backend_config" in metafunc.fixturenames:
        vertexai_setting = os.environ.get("TEST_VERTEXAI", "").lower()

        if vertexai_setting == "only":
            # Test only Vertex AI
            backends = ["vertex_ai"]
        elif vertexai_setting in ("1", "true", "yes"):
            # Test both backends
            backends = ["google_ai", "vertex_ai"]
        else:
            # Default: test only Google AI
            backends = ["google_ai"]

        metafunc.parametrize("backend_config", backends, indirect=True)


def _is_quota_exhausted_error(error: BaseException) -> bool:
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    if call.excinfo and _is_quota_exhausted_error(call.excinfo.value):
        report.outcome = "skipped"
        report.wasxfail = "Skipped due to Gemini API quota exhaustion"


@pytest.fixture
def backend_config(request: pytest.FixtureRequest) -> dict:
    """Provide backend configuration.

    Args:
        request: The pytest fixture request object.

    Returns:
        Backend configuration dictionary to pass to ChatGoogleGenerativeAI.
    """
    if request.param == "vertex_ai":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            pytest.skip("Vertex AI requires GOOGLE_CLOUD_PROJECT env var")
        return {"vertexai": True, "project": project, "api_key": None}
    # Google AI backend (default)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Google AI requires GOOGLE_API_KEY or GEMINI_API_KEY env var")
    return {"api_key": api_key}
