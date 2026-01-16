"""Tests configuration to be executed before tests execution."""

from collections.abc import Generator
from typing import Any

import pytest

_RELEASE_FLAG = "release"
_GPU_FLAG = "gpu"
_LONG_FLAG = "long"
_EXTENDED_FLAG = "extended"

_PYTEST_FLAGS = [_RELEASE_FLAG, _GPU_FLAG, _LONG_FLAG, _EXTENDED_FLAG, "first"]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add flags accepted by pytest CLI.

    Args:
        parser: The pytest parser object.

    Returns:

    """
    for flag in _PYTEST_FLAGS:
        parser.addoption(
            f"--{flag}", action="store_true", default=False, help=f"run {flag} tests"
        )


def pytest_configure(config: pytest.Config) -> None:
    """Add pytest custom configuration.

    Args:
        config: The pytest config object.

    Returns:
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

    Returns:
    """
    for item in items:
        keywords = list(set(item.keywords).intersection(_PYTEST_FLAGS))
        if keywords and not any(config.getoption(f"--{kw}") for kw in keywords):
            skip = pytest.mark.skip(reason=f"need --{keywords[0]} option to run")
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[None, Any, None]:
    yield
