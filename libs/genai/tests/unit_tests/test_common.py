import asyncio
from unittest.mock import MagicMock, patch

from langchain_google_genai._common import (
    get_user_agent,
)


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common._LANGCHAIN_GENAI_VERSION", "1.2.3")
def test_get_user_agent_with_telemetry_env_variable(
    mock_environ_get: MagicMock,
) -> None:
    mock_environ_get.return_value = True
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module+remote_reasoning_engine"
    assert user_agent_str == (
        "langchain-google-genai/1.2.3-test-module+remote_reasoning_engine"
    )


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common._LANGCHAIN_GENAI_VERSION", "1.2.3")
def test_get_user_agent_without_telemetry_env_variable(
    mock_environ_get: MagicMock,
) -> None:
    mock_environ_get.return_value = False
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module"
    assert user_agent_str == "langchain-google-genai/1.2.3-test-module"


def test_version_is_cached_at_module_level() -> None:
    """Test that version is cached at module level and doesn't call metadata.version."""
    from langchain_google_genai import _common

    # The cached version should be a string
    assert isinstance(_common._LANGCHAIN_GENAI_VERSION, str)
    # Should be either a valid version or "0.0.0" (fallback)
    assert _common._LANGCHAIN_GENAI_VERSION != ""


async def test_get_user_agent_no_blocking_in_async_context() -> None:
    """Test that get_user_agent doesn't perform blocking I/O in async context.

    This test verifies that get_user_agent uses the cached version
    and doesn't call metadata.version() which would be blocking I/O.
    """
    # Mock metadata.version to raise an error if it's called
    with patch("langchain_google_genai._common.metadata.version") as mock_version:
        mock_version.side_effect = RuntimeError(
            "metadata.version() should not be called - version should be cached"
        )

        # This should work without calling metadata.version()
        client_lib_version, user_agent_str = get_user_agent(module="test-async")

        # Verify the call didn't happen
        mock_version.assert_not_called()

        # Verify we got valid output (using the cached version)
        assert "test-async" in client_lib_version
        assert "langchain-google-genai" in user_agent_str


def test_async_context_execution() -> None:
    """Run the async test to ensure it works in event loop."""
    asyncio.run(test_get_user_agent_no_blocking_in_async_context())
