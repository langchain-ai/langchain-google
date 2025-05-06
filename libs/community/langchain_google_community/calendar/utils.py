"""Google Calendar tool utils."""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_core.utils import guard_import

from langchain_google_community._utils import (
    get_google_credentials,
    import_googleapiclient_resource_builder,
)

if TYPE_CHECKING:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import Resource
    from googleapiclient.discovery import build as build_resource

logger = logging.getLogger(__name__)


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def build_calendar_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "calendar",
    service_version: str = "v3",
) -> Resource:
    """Build a Google Calendar service."""
    credentials = credentials or get_google_credentials(scopes=DEFAULT_SCOPES)
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)


def build_resouce_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "calendar",
    service_version: str = "v3",
) -> Resource:
    warnings.warn(
        "build_resource_service is deprecated and will be removed in a future version."
        "Use build_calendar_service instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_calendar_service(credentials, service_name, service_version)


def is_all_day_event(start_datetime: str, end_datetime: str) -> bool:
    """Check if the event is all day."""
    try:
        datetime.strptime(start_datetime, "%Y-%m-%d")
        datetime.strptime(end_datetime, "%Y-%m-%d")
        return True
    except ValueError:
        return False
