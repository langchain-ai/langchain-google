"""Gmail tool utils."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, List, Optional

from langchain_google_community._utils import (
    get_google_credentials,
    import_googleapiclient_resource_builder,
)

if TYPE_CHECKING:
    from email.message import EmailMessage

    from google.oauth2.credentials import Credentials  # type: ignore[import]
    from googleapiclient.discovery import Resource  # type: ignore[import]

logger = logging.getLogger(__name__)


DEFAULT_SCOPES = ["https://mail.google.com/"]
DEFAULT_SERVICE_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_credentials(
    token_file: Optional[str] = None,
    client_sercret_file: Optional[str] = None,
    service_account_file: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    use_domain_wide: bool = False,
    delegated_user: Optional[str] = None,
) -> Credentials:
    """Get Gmail credentials."""
    warnings.warn(
        "get_gmail_credentials is deprecated and will be removed in a future version."
        "Use get_google_credentials instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if use_domain_wide:
        scopes = scopes or DEFAULT_SERVICE_SCOPES
    else:
        scopes = scopes or DEFAULT_SCOPES

    return get_google_credentials(
        scopes=scopes,
        token_file=token_file,
        client_secrets_file=client_sercret_file,
        service_account_file=service_account_file,
        use_domain_wide=use_domain_wide,
        delegated_user=delegated_user,
    )


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
        scopes = scopes or DEFAULT_SERVICE_SCOPES
    else:
        scopes = scopes or DEFAULT_SCOPES

    credentials = credentials or get_google_credentials(
        scopes=scopes,
        use_domain_wide=use_domain_wide,
        delegated_user=delegated_user,
        service_account_file=service_account_file,
    )
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)


def build_resource_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "gmail",
    service_version: str = "v1",
    use_domain_wide: bool = False,
    delegated_user: Optional[str] = None,
    service_account_file: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Resource:
    """Build a Gmail resource service."""
    warnings.warn(
        "build_resource_service is deprecated and will be removed in a future version."
        "Use build_gmail_service instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_gmail_service(
        credentials=credentials,
        service_name=service_name,
        service_version=service_version,
        use_domain_wide=use_domain_wide,
        delegated_user=delegated_user,
        service_account_file=service_account_file,
        scopes=scopes,
    )


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


def get_email_body(message: EmailMessage) -> str:
    """Get the text of an email's body.

    The body is the plain text part, or else the HTML part as text: an HTML email
    that carries attachments or inline images often has no plain part. It is
    decoded with the charset the email declares.

    Args:
        message: The email, parsed with `email.policy.default`.

    Returns:
        The text of the body, or an empty string when the email has none.
    """
    body = message.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    try:
        content = body.get_content()
    except LookupError:
        # A charset that Python does not know.
        content = body.get_payload(decode=True).decode("utf-8", errors="replace")  # type: ignore[union-attr]
    if body.get_content_type() == "text/html":
        return clean_email_body(content)
    return content
