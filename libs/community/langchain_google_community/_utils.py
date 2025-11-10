"""Utilities to init Vertex AI."""

from __future__ import annotations

import logging
import os
from importlib import metadata
from typing import TYPE_CHECKING, List, Optional, Tuple

from google.api_core.gapic_v1.client_info import ClientInfo
from langchain_core.utils import guard_import

_TELEMETRY_TAG = "remote_reasoning_engine"
_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

if TYPE_CHECKING:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build as build_resource

logger = logging.getLogger(__name__)


def get_user_agent(module: Optional[str] = None) -> Tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module: The module for a custom user agent header.
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
        module: The module for a custom user agent header.
    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=user_agent,
    )


def import_google() -> Tuple[Request, Credentials, ServiceCredentials]:
    """Import google libraries.

    Returns:
        `Request` and `Credentials` classes.
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
        ).Credentials,
    )


def import_installed_app_flow() -> InstalledAppFlow:
    """Import `InstalledAppFlow` class.

    Returns:
        `InstalledAppFlow` class.
    """
    return guard_import(
        module_name="google_auth_oauthlib.flow", pip_name="google-auth-oauthlib"
    ).InstalledAppFlow


def import_googleapiclient_resource_builder() -> build_resource:
    """Import `googleapiclient.discovery.build` function.

    Returns:
        `googleapiclient.discovery.build` function.
    """
    return guard_import(
        module_name="googleapiclient.discovery", pip_name="google-api-python-client"
    ).build


DEFAULT_CREDS_TOKEN_FILE = "token.json"
DEFAULT_CLIENT_SECRETS_FILE = "credentials.json"
DEFAULT_SERVICE_ACCOUNT_FILE = "service_account.json"


def get_google_credentials(
    scopes: List[str],
    token_file: Optional[str] = None,
    client_secrets_file: Optional[str] = None,
    service_account_file: Optional[str] = None,
    use_domain_wide: bool = False,
    delegated_user: Optional[str] = None,
) -> Credentials:
    """Get credentials."""
    if use_domain_wide:
        _, _, ServiceCredentials = import_google()
        service_account_file = service_account_file or DEFAULT_SERVICE_ACCOUNT_FILE
        credentials = ServiceCredentials.from_service_account_file(
            service_account_file, scopes=scopes
        )

        if delegated_user:
            credentials = credentials.with_subject(delegated_user)

        return credentials
    else:
        Request, Credentials, ServiceCredentials = import_google()
        InstalledAppFlow = import_installed_app_flow()
        creds = None
        token_file = token_file or DEFAULT_CREDS_TOKEN_FILE
        client_secrets_file = client_secrets_file or DEFAULT_CLIENT_SECRETS_FILE
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())  # type: ignore[call-arg]
            else:
                # https://developers.google.com/calendar/api/quickstart/python#authorize_credentials_for_a_desktop_application # noqa
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secrets_file, scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_file, "w") as token:
                token.write(creds.to_json())
        return creds
