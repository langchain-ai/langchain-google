from unittest.mock import ANY, MagicMock, patch

import pytest

from langchain_google_community.gemini_files import register_gcs_files


def test_register_gcs_files_success() -> None:
    """Test successful registration of GCS files."""
    mock_uris = ["gs://bucket/file1.png", "gs://bucket/file2.pdf"]
    mock_files = [MagicMock(name="file1"), MagicMock(name="file2")]

    with patch("google.genai.Client") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.files.register_files.return_value = MagicMock(files=mock_files)

        with patch("google.auth.default", return_value=(MagicMock(), "project-id")):
            result = register_gcs_files(
                uris=mock_uris, project="test-project", api_key="test-api-key"
            )

            assert result == mock_files
            mock_client_class.assert_called_once_with(
                api_key="test-api-key",
                project="test-project",
                location=None,
                credentials=ANY,
            )
            mock_client.files.register_files.assert_called_once_with(
                uris=mock_uris,
                auth=ANY,
            )


def test_register_gcs_files_import_error() -> None:
    """Test ImportError when google-genai is missing."""
    with patch("google.genai.Client", side_effect=ImportError):
        # Mock get_from_dict_or_env to return a dummy key if not provided
        with patch(
            "langchain_google_community.gemini_files.get_from_dict_or_env",
            return_value="dummy-key",
        ):
            with patch("google.auth.default", return_value=(MagicMock(), "project-id")):
                with pytest.raises(ImportError) as exc_info:
                    register_gcs_files(uris=["gs://bucket/file"])
                assert "google-genai" in str(exc_info.value)
