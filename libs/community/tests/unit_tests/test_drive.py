from pathlib import Path
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


def _make_loader_with_writable_paths(tmp_path: Path) -> GoogleDriveLoader:
    """Build a loader whose service_account_key / token_path do NOT exist
    and whose credentials_path exists (so the field validator passes)."""
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text("{}")
    return GoogleDriveLoader(
        folder_id="dummy_folder",
        credentials_path=creds_file,
        token_path=tmp_path / "token.json",
        service_account_key=tmp_path / "service-account.json",
    )


def test_load_credentials_uses_installed_app_flow_when_adc_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Desktop OAuth path: with `GOOGLE_APPLICATION_CREDENTIALS` unset and a
    `credentials_path` supplied, `_load_credentials` must drive `InstalledAppFlow`
    and must NOT fall back to `google.auth.default()` (which requires ADC and
    would crash for desktop OAuth users -- regression test for the inverted
    `not in os.environ` check)."""
    pytest.importorskip("google_auth_oauthlib")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    loader = _make_loader_with_writable_paths(tmp_path)

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = "{}"
    fake_flow = MagicMock()
    fake_flow.run_local_server.return_value = fake_creds

    with (
        patch(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
            return_value=fake_flow,
        ) as mock_from_file,
        patch("google.auth.default") as mock_default,
    ):
        result = loader._load_credentials()

    mock_from_file.assert_called_once()
    mock_default.assert_not_called()
    assert result is fake_creds
    assert loader.token_path.exists()


def test_load_credentials_uses_default_when_adc_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADC path: with `GOOGLE_APPLICATION_CREDENTIALS` set, `_load_credentials`
    must use `google.auth.default()` and must NOT trigger the InstalledAppFlow
    browser-OAuth dance."""
    pytest.importorskip("google_auth_oauthlib")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "adc.json"))
    loader = _make_loader_with_writable_paths(tmp_path)

    fake_default_creds = MagicMock()
    fake_scoped_creds = MagicMock()

    with (
        patch(
            "google.auth.default",
            return_value=(fake_default_creds, "test-project"),
        ) as mock_default,
        patch(
            "google.auth.credentials.with_scopes_if_required",
            return_value=fake_scoped_creds,
        ),
        patch(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
        ) as mock_flow,
    ):
        result = loader._load_credentials()

    mock_default.assert_called_once()
    mock_flow.assert_not_called()
    assert result is fake_scoped_creds
