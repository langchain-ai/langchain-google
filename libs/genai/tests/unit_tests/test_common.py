from unittest.mock import MagicMock, patch

from langchain_google_genai._common import (
    get_user_agent,
)


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.version")
def test_get_user_agent_with_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = True
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module+remote_reasoning_engine"
    assert user_agent_str == (
        "langchain-google-genai/1.2.3-test-module+remote_reasoning_engine"
    )


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.version")
def test_get_user_agent_without_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = False
    client_lib_version, user_agent_str = get_user_agent(module="test-module")
    assert client_lib_version == "1.2.3-test-module"
    assert user_agent_str == "langchain-google-genai/1.2.3-test-module"
