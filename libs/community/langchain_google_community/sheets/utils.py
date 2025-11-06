"""Google Sheets tool utils."""

from __future__ import annotations

import re
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
    # Default scopes for full access (read/write)
    # Note: Use scopes parameter to override if read-only access is desired
    default_scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
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

    !!! note

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
        The validated spreadsheet ID.

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


# These patterns and validators are actively used by all read/write tools
# to ensure proper A1 notation format before making API calls.

# A1 notation regex patterns
# Single cell: A1, Z99 (column letters + row starting with 1-9)
_CELL = re.compile(r"^[A-Za-z]+[1-9]\d*$")
# Area: A1:B2 (two cells separated by colon)
_AREA = re.compile(r"^[A-Za-z]+[1-9]\d*:[A-Za-z]+[1-9]\d*$")
# Whole columns: A:A, B:D (column letters on both sides)
_COLS = re.compile(r"^[A-Za-z]+:[A-Za-z]+$")
# Whole rows: 1:1, 5:10 (row numbers starting with 1-9 on both sides)
_ROWS = re.compile(r"^[1-9]\d*:[1-9]\d*$")
# Named range: MyData, Sales_2024 (alphanumeric + underscore, letter/underscore start)
_NAMED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_target(target: str) -> bool:
    """Validate a range target (after sheet qualifier removed).

    Args:
        target: Range string without sheet qualifier.

    Returns:
        True if valid, False otherwise.
    """
    # Check specific patterns first (more restrictive)
    if _AREA.fullmatch(target):
        return True
    if _CELL.fullmatch(target):
        return True
    if _COLS.fullmatch(target):
        return True
    if _ROWS.fullmatch(target):
        return True
    # Check named range last (least restrictive)
    # Only allow if it looks like a name, not like a malformed cell reference
    if _NAMED.fullmatch(target):
        # Reject simple cell-ish names that end with 0 (e.g., "A0")
        if target[-1] == "0" and any(c.isalpha() for c in target):
            return False
        return True
    return False


def validate_a1_range(range_name: str) -> bool:
    """Validate A1 notation range format.

    Supports:
    - Single cells: `'A1'`, `'Z99'`
    - Areas: `'A1:B2'`
    - Whole columns: `'A:A'`, `'B:D'`
    - Whole rows: `'1:1'`, `'5:10'`
    - Sheet-qualified: `'Sheet1!A1'`, `"'My Sheet'!A1:B2"`
    - Named ranges: `'MyData'`, `'Sales_2024'`

    Args:
        range_name: Range string to validate.

    Returns:
        `True` if valid format, `False` otherwise.

    Examples:
        >>> validate_a1_range("A1")
        True
        >>> validate_a1_range("A1:B2")
        True
        >>> validate_a1_range("Sheet1!A1")
        True
        >>> validate_a1_range("A:A")
        True
        >>> validate_a1_range("")
        False
    """
    if not range_name or not range_name.strip():
        return False
    if range_name.startswith("!"):
        return False

    sheet, had_bang, after = range_name.partition("!")
    if had_bang:
        if not after:  # "Sheet1!" -> invalid
            return False
        target = after
    else:
        target = range_name

    return _validate_target(target)


def validate_range_name(range_name: str) -> str:
    """Validate and normalize a range name.

    Permissive A1 validator. Supports:
    - Single cells: `'A1'`, `'Z99'`
    - Areas: `'A1:B2'`
    - Whole columns: `'A:A'`, `'B:D'`
    - Whole rows: `'1:1'`, `'5:10'`
    - Sheet-qualified: `'Sheet1!A1'`, `"'My Sheet'!A1:B2"`
    - Named ranges: `'MyData'`, `'Sales_2024'`

    Args:
        range_name: The range name to validate (e.g., `'A1:Z100'`, `'Sheet1!A1:B2'`).

    Returns:
        The validated range name.

    Raises:
        ValueError: If the range name is invalid.

    Examples:
        >>> validate_range_name("A1")
        'A1'
        >>> validate_range_name("Sheet1!A1:B2")
        'Sheet1!A1:B2'
        >>> validate_range_name("A:A")
        'A:A'
        >>> validate_range_name("")
        Traceback (most recent call last):
        ...
        ValueError: Invalid range format: ...
    """
    if not validate_a1_range(range_name):
        raise ValueError(
            f"Invalid range format: {range_name}. "
            "Expected A1 notation like 'A1', 'A1:B2', 'Sheet1!A1', 'A:A', or '1:1'. "
            "Named ranges are allowed."
        )

    return range_name
