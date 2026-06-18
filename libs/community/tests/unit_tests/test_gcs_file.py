"""Unit tests for the GCS file loader."""

from unittest.mock import patch

import pytest

from langchain_google_community.gcs_file import GCSFileLoader


def test_gcs_file_loader_default_loader_missing_langchain_community() -> None:
    """The default unstructured loader should explain its optional dependency."""
    loader = GCSFileLoader(
        project_name="project",
        bucket="bucket",
        blob="blob.txt",
    )

    original_import = __import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "langchain_community.document_loaders.unstructured":
            raise ImportError("No module named 'langchain_community'")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ImportError) as excinfo,
    ):
        loader._loader_func("/tmp/blob.txt")

    assert "Either provide a custom loader with loader_func argument" in str(
        excinfo.value
    )
    assert "`pip install langchain-community`" in str(excinfo.value)
