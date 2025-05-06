"""Gmail tool utils."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_google_community._utils import (
    get_google_credentials,
    import_googleapiclient_resource_builder,
)

if TYPE_CHECKING:
    from google.auth.transport.requests import Request  # type: ignore[import]
    from google.oauth2.credentials import Credentials  # type: ignore[import]
    from google.oauth2.service_account import Credentials as ServiceCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import]
    from googleapiclient.discovery import Resource  # type: ignore[import]
    from googleapiclient.discovery import build as build_resource

logger = logging.getLogger(__name__)


DEFAULT_SCOPES = ["https://mail.google.com/"]
DEFAULT_SERVICE_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def build_gmail_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "gmail",
    service_version: str = "v1",
    use_domain_wide: bool = False,
    delegated_user: Optional[str] = None,
    service_account_file: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Resource:
    """Build a Gmail service."""
    if use_domain_wide:
        credentials = credentials or get_google_credentials(
            scopes=scopes or DEFAULT_SERVICE_SCOPES,
            use_domain_wide=use_domain_wide,
            delegated_user=delegated_user,
            service_account_file=service_account_file,
        )
    else:
        credentials = credentials or get_google_credentials(
            scopes=scopes or DEFAULT_SCOPES,
            use_domain_wide=use_domain_wide,
            delegated_user=delegated_user,
            service_account_file=service_account_file,
        )
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)


def clean_email_body(body: str) -> str:
    """Clean email body."""
    try:
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(str(body), "html.parser")
            body = soup.get_text()
            return str(body)
        except Exception as e:
            logger.error(e)
            return str(body)
    except ImportError:
        logger.warning("BeautifulSoup not installed. Skipping cleaning.")
        return str(body)
