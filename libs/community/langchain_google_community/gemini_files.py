"""Utilities for working with the Gemini Files API."""

from typing import Any, Optional

import google.auth
from langchain_core.utils import get_from_dict_or_env


def register_gcs_files(
    uris: list[str],
    project: Optional[str] = None,
    location: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[Any]:
    """Register GCS files with the Gemini Files API.

    This utility allows you to use files already stored in Google Cloud Storage
    directly with the Gemini API without having to upload them again.

    Args:
        uris: A list of URIs starting with 'gs://'.
        project: Google Cloud project ID. If not provided, will be ascertained
            from the environment.
        location: Google Cloud location (e.g., 'us-central1').
        api_key: Gemini API key. If not provided, will be searched for in
            environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).

    Returns:
        A list of registered file objects from the Gemini API.

    Raises:
        ImportError: If 'google-genai' package is not installed.
        ValueError: If no API key or project is found.
    """
    msg = (
        "Could not import google-genai python package. "
        "Please install it with `pip install google-genai`."
    )
    try:
        from google.genai import Client
    except ImportError as e:
        raise ImportError(msg) from e

    # Resolve API Key
    resolved_api_key = (
        api_key
        or get_from_dict_or_env({}, "api_key", "GOOGLE_API_KEY", default=None)
        or get_from_dict_or_env({}, "api_key", "GEMINI_API_KEY", default=None)
    )

    # Authenticate with required scopes
    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.read_only",
    ]
    credentials, _ = google.auth.default(scopes=scopes)

    try:
        client = Client(
            api_key=resolved_api_key,
            project=project,
            location=location,
            credentials=credentials,
        )
    except ImportError as e:
        raise ImportError(msg) from e

    # Register the files
    response = client.files.register_files(  # type: ignore[attr-defined]
        uris=uris,
        auth=None,
    )
    return response.files or []
