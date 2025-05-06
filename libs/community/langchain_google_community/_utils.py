"""Utilities to init Vertex AI."""

import os
from importlib import metadata
from typing import Optional, Tuple
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_core.utils import guard_import
from google.api_core.gapic_v1.client_info import ClientInfo

_TELEMETRY_TAG = "remote_reasoning_engine"
_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

if TYPE_CHECKING:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import Resource
    from googleapiclient.discovery import build as build_resource

logger = logging.getLogger(__name__)

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
    if os.environ.get(_TELEMETRY_ENV_VARIABLE_NAME):
        client_library_version += f"+{_TELEMETRY_TAG}"
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


def import_google() -> Tuple[Request, Credentials, ServiceCredentials]:
    """Import google libraries.

    Returns:
        Tuple[Request, Credentials]: Request and Credentials classes.
    """
    return (
        guard_import(
            module_name="google.auth.transport.requests",
            pip_name="google-auth",
        ).Request,
        guard_import(
            module_name="google.oauth2.credentials", pip_name="google-auth"
        ).Credentials,
        guard_import(
            module_name="google.oauth2.service_account", pip_name="google-auth"
        ).ServiceCredentials
    )