"""Unit tests for the BigQueryLoader."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_google_community.bigquery import BigQueryLoader


@patch("langchain_google_community.bigquery.import_bigquery")
def test_bigquery_loader_import_error(mock_import_bigquery: MagicMock) -> None:
    """Test that BigQueryLoader raises an ImportError if
    google-cloud-bigquery is not installed."""
    # Simulate the ImportError that occurs when the dependency is missing
    mock_import_bigquery.side_effect = ImportError(
        "Could not import google.cloud.bigquery python package. "
        "Please, install it with `pip install langchain-google-community[bigquery]`"
    )

    with pytest.raises(ImportError) as excinfo:
        BigQueryLoader(query="SELECT 1")

    assert "Could not import google.cloud.bigquery python package." in str(
        excinfo.value
    )
    assert (
        "Please, install it with `pip install langchain-google-community[bigquery]`"
        in str(excinfo.value)
    )
