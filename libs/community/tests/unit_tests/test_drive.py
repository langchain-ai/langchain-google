from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

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


def _make_mock_service(mime_type: str, file_id: str) -> MagicMock:
    """Return a mock Drive service whose files().list() returns one file entry."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {
        "files": [{"id": file_id, "mimeType": mime_type}]
    }
    return mock_service


def test_load_documents_from_ids_dispatches_sheets() -> None:
    """Spreadsheet IDs must be routed to _load_sheet_from_id."""
    sheet_doc = Document(page_content="row1", metadata={})
    loader = GoogleDriveLoader(document_ids=["sheet_id_123"])
    with (
        patch.object(loader, "_load_credentials"),
        patch(
            "googleapiclient.discovery.build",
            return_value=_make_mock_service(
                "application/vnd.google-apps.spreadsheet", "sheet_id_123"
            ),
        ),
        patch.object(
            loader, "_load_sheet_from_id", return_value=[sheet_doc]
        ) as mock_sheet,
        patch.object(loader, "_load_document_from_id") as mock_doc,
    ):
        result = loader._load_documents_from_ids()

    mock_sheet.assert_called_once_with("sheet_id_123")
    mock_doc.assert_not_called()
    assert result == [sheet_doc]


def test_load_documents_from_ids_dispatches_docs() -> None:
    """Google Doc IDs must be routed to _load_document_from_id."""
    doc = Document(page_content="hello", metadata={})
    loader = GoogleDriveLoader(document_ids=["doc_id_456"])
    with (
        patch.object(loader, "_load_credentials"),
        patch(
            "googleapiclient.discovery.build",
            return_value=_make_mock_service(
                "application/vnd.google-apps.document", "doc_id_456"
            ),
        ),
        patch.object(loader, "_load_sheet_from_id") as mock_sheet,
        patch.object(loader, "_load_document_from_id", return_value=doc) as mock_doc,
    ):
        result = loader._load_documents_from_ids()

    mock_doc.assert_called_once_with("doc_id_456")
    mock_sheet.assert_not_called()
    assert result == [doc]


def test_load_documents_from_ids_dispatches_pdfs() -> None:
    """PDF IDs must be routed to _load_file_from_id, not _load_document_from_id."""
    pdf_doc = Document(page_content="pdf content", metadata={})
    loader = GoogleDriveLoader(document_ids=["pdf_id_789"])
    with (
        patch.object(loader, "_load_credentials"),
        patch(
            "googleapiclient.discovery.build",
            return_value=_make_mock_service("application/pdf", "pdf_id_789"),
        ),
        patch.object(loader, "_load_sheet_from_id") as mock_sheet,
        patch.object(loader, "_load_document_from_id") as mock_doc,
        patch.object(loader, "_load_file_from_id", return_value=[pdf_doc]) as mock_file,
    ):
        result = loader._load_documents_from_ids()

    mock_file.assert_called_once_with("pdf_id_789")
    mock_sheet.assert_not_called()
    mock_doc.assert_not_called()
    assert result == [pdf_doc]
