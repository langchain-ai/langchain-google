from unittest.mock import MagicMock, patch

from google.api_core.gapic_v1.client_info import ClientInfo

from langchain_google_genai._common import (
    get_client_info,
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


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.version")
def test_get_client_info_with_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = True
    client_info = get_client_info(module="test-module")
    assert isinstance(client_info, ClientInfo)
    assert client_info.client_library_version == "1.2.3-test-module+remote_reasoning_engine"
    assert client_info.gapic_version == "1.2.3-test-module+remote_reasoning_engine"
    assert client_info.user_agent == (
        "langchain-google-genai/1.2.3-test-module+remote_reasoning_engine"
    )


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.version")
def test_get_client_info_without_telemetry_env_variable(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = False
    client_info = get_client_info(module="test-module")
    assert isinstance(client_info, ClientInfo)
    assert client_info.client_library_version == "1.2.3-test-module"
    assert client_info.gapic_version == "1.2.3-test-module"
    assert client_info.user_agent == "langchain-google-genai/1.2.3-test-module"


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.version")
def test_get_client_info_no_module(
    mock_version: MagicMock, mock_environ_get: MagicMock
) -> None:
    mock_version.return_value = "1.2.3"
    mock_environ_get.return_value = False
    client_info = get_client_info()
    assert isinstance(client_info, ClientInfo)
    assert client_info.client_library_version == "1.2.3"
    assert client_info.gapic_version == "1.2.3"
    assert client_info.user_agent == "langchain-google-genai/1.2.3"


@patch("langchain_google_genai._common.os.environ.get")
@patch("langchain_google_genai._common.metadata.PackageNotFoundError")
@patch("langchain_google_genai._common.metadata.version")
def test_get_client_info_package_not_found(
    mock_version: MagicMock,
    mock_package_not_found_error: MagicMock,
    mock_environ_get: MagicMock,
) -> None:
    mock_version.side_effect = mock_package_not_found_error
    mock_environ_get.return_value = False
    client_info = get_client_info(module="test-module")
    assert isinstance(client_info, ClientInfo)
    assert client_info.client_library_version == "0.0.0-test-module"
    assert client_info.gapic_version == "0.0.0-test-module"
    assert client_info.user_agent == "langchain-google-genai/0.0.0-test-module"
