"""Utilities to init Vertex AI."""

from importlib import metadata
import logging
from typing import Optional, Tuple

from google.api_core.gapic_v1.client_info import ClientInfo

try:
    from google.cloud.aiplatform import telemetry
except ModuleNotFoundError as e:
    telemetry = None
    logging.debug(
        "Cannot import telemetry to add custom tool context."
        "Please run `pip install google-cloud-aiplatform`."
        f"Error: {e}"
    )


def get_user_agent(module: Optional[str] = None) -> Tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        Tuple[str, str]
    """
    try:
        langchain_version = metadata.version("langchain-google-community")
    except metadata.PackageNotFoundError:
        langchain_version = "0.0.0"
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    if telemetry and telemetry._tool_names_to_append:
        client_library_version += f"+tools+{'+'.join(telemetry._tool_names_to_append[::-1])}"
    return (
        client_library_version,
        f"langchain-google-community/{client_library_version}",
    )


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""Returns a client info object with a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=user_agent,
    )
