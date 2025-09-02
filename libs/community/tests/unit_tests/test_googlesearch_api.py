"""Integration test for Google Search API Wrapper."""

from unittest.mock import patch

import pytest

from langchain_google_community.search import GoogleSearchAPIWrapper


@pytest.mark.extended
def test_no_result_call() -> None:
    """Test that call gives no result."""
    with patch("googleapiclient.discovery.build") as search_engine:
        search_engine.return_value.cse.return_value.list.return_value.execute.return_value = {}  # noqa: E501
        search = GoogleSearchAPIWrapper(  # type: ignore[call-arg]
            google_api_key="key", google_cse_id="cse"
        )
        output = search.run("test")
        search_engine.assert_called_once_with("customsearch", "v1", developerKey="key")
        assert "No good Google Search Result was found" == output
