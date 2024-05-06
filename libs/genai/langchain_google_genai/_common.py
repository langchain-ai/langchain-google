from importlib import metadata
from typing import Optional, Tuple, TypedDict

from google.api_core.gapic_v1.client_info import ClientInfo

from langchain_google_genai._enums import HarmBlockThreshold, HarmCategory


class GoogleGenerativeAIError(Exception):
    """
    Custom exception class for errors associated with the `Google GenAI` API.
    """


def get_user_agent(module: Optional[str] = None) -> Tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        Tuple[str, str]
    """
    try:
        langchain_version = metadata.version("langchain-google-genai")
    except metadata.PackageNotFoundError:
        langchain_version = "0.0.0"
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    return client_library_version, f"langchain-google-genai/{client_library_version}"


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


class SafetySettingDict(TypedDict):
    category: HarmCategory
    threshold: HarmBlockThreshold
