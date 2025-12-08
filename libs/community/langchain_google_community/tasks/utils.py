"""Google Tasks tool utils."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from langchain_google_community._utils import (
    get_google_credentials,
    import_googleapiclient_resource_builder,
)

if TYPE_CHECKING:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import Resource

logger = logging.getLogger(__name__)


DEFAULT_SCOPES = ["https://www.googleapis.com/auth/tasks"]


def build_tasks_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "tasks",
    service_version: str = "v1",
) -> Resource:
    """Build a Google Tasks service.

    Args:
        credentials: Optional credentials to use. If not provided,
            will use default credentials.
        service_name: The name of the service. Default is 'tasks'.
        service_version: The version of the service. Default is 'v1'.

    Returns:
        A Resource object for interacting with the Google Tasks API.
    """
    credentials = credentials or get_google_credentials(scopes=DEFAULT_SCOPES)
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)
