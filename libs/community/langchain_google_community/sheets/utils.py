"""Google Sheets tool utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain_google_community._utils import (
    get_google_credentials,
    import_googleapiclient_resource_builder,
)

if TYPE_CHECKING:
    from google.oauth2.credentials import Credentials  # type: ignore[import]
    from googleapiclient.discovery import Resource  # type: ignore[import]


def build_sheets_service(
    credentials: Optional[Credentials] = None,
    service_name: str = "sheets",
    service_version: str = "v4",
    use_domain_wide: bool = False,
    delegated_user: Optional[str] = None,
    service_account_file: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> Resource:
    """Build a Google Sheets service with OAuth2 credentials (full access).

    Args:
        credentials: OAuth2 credentials. If None, will attempt to load from
            default locations.
        service_name: The Google API service name.
        service_version: The Google API service version.
        use_domain_wide: Whether to use domain-wide delegation.
        delegated_user: User to impersonate for domain-wide delegation.
        service_account_file: Path to service account file for domain-wide
            delegation.
        scopes: List of OAuth2 scopes. Defaults to full access scopes.

    Returns:
        Resource: Google Sheets API service with full access capabilities.
    """
    # Default scopes for full access
    default_scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if use_domain_wide:
        # Read-only scopes for service accounts
        default_scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]

    scopes = scopes or default_scopes

    credentials = credentials or get_google_credentials(
        scopes=scopes,
        use_domain_wide=use_domain_wide,
        delegated_user=delegated_user,
        service_account_file=service_account_file,
    )
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)


def build_sheets_service_with_api_key(api_key: str) -> Resource:
    """Build a Google Sheets service with API key (read-only access).

    Args:
        api_key: Google API key for authentication.

    Returns:
        Resource: Google Sheets API service with read-only access to public
            spreadsheets.

    Note:
        API key authentication only works with public spreadsheets.
        For private spreadsheets, use OAuth2 credentials instead.
    """
    builder = import_googleapiclient_resource_builder()
    return builder("sheets", "v4", developerKey=api_key)


def validate_spreadsheet_id(spreadsheet_id: str) -> str:
    """Validate and normalize a Google Spreadsheet ID.

    Args:
        spreadsheet_id: The spreadsheet ID to validate.

    Returns:
        str: The validated spreadsheet ID.

    Raises:
        ValueError: If the spreadsheet ID is invalid.
    """
    if not spreadsheet_id:
        raise ValueError("Spreadsheet ID cannot be empty")

    # Remove any URL components if present
    if "docs.google.com/spreadsheets/d/" in spreadsheet_id:
        # Extract ID from URL
        start = spreadsheet_id.find("/d/") + 3
        end = spreadsheet_id.find("/", start)
        if end == -1:
            end = len(spreadsheet_id)
        spreadsheet_id = spreadsheet_id[start:end]

    # Basic validation - Google Spreadsheet IDs are typically 44 characters
    if len(spreadsheet_id) < 20:
        raise ValueError(f"Invalid spreadsheet ID format: {spreadsheet_id}")

    return spreadsheet_id


def validate_range_name(range_name: str) -> str:
    """Validate and normalize a range name.

    Args:
        range_name: The range name to validate (e.g., "A1:Z100", "Sheet1!A1:B2").

    Returns:
        str: The validated range name.

    Raises:
        ValueError: If the range name is invalid.
    """
    if not range_name:
        raise ValueError("Range name cannot be empty")

    # Basic validation - should contain at least one colon for range
    if ":" not in range_name and not range_name.isalpha():
        raise ValueError(f"Invalid range format: {range_name}")

    return range_name
