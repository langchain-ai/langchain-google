import pytest

from langchain_google_community.drive import GoogleDriveLoader


def test_drive_default_scope() -> None:
    """Test that default scope is set correctly."""
    loader = GoogleDriveLoader(folder_id="dummy_folder")
    assert loader.scopes == ["https://www.googleapis.com/auth/drive.file"]


def test_drive_custom_scope() -> None:
    """Test setting custom scope."""
    custom_scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    loader = GoogleDriveLoader(folder_id="dummy_folder", scopes=custom_scopes)
    assert loader.scopes == custom_scopes


def test_drive_multiple_scopes() -> None:
    """Test setting multiple valid scopes."""
    custom_scopes = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.metadata.readonly",
    ]
    loader = GoogleDriveLoader(folder_id="dummy_folder", scopes=custom_scopes)
    assert loader.scopes == custom_scopes


def test_drive_empty_scope_list() -> None:
    """Test that empty scope list raises error."""
    with pytest.raises(ValueError, match="At least one scope must be provided"):
        GoogleDriveLoader(folder_id="dummy_folder", scopes=[])


def test_drive_invalid_scope() -> None:
    """Test that invalid scope raises error."""
    invalid_scopes = ["https://www.googleapis.com/auth/drive.invalid"]
    with pytest.raises(ValueError, match="Invalid Google Drive API scope"):
        GoogleDriveLoader(folder_id="dummy_folder", scopes=invalid_scopes)
