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


def test_results_includes_image_and_thumbnail() -> None:
    """Test that results extracts image and thumbnail from pagemap."""
    with patch("googleapiclient.discovery.build") as search_engine:
        mock_execute = search_engine.return_value.cse
        mock_execute = mock_execute.return_value.list
        mock_execute = mock_execute.return_value.execute
        mock_execute.return_value = {
            "items": [
                {
                    "title": "Test Title",
                    "link": "https://example.com",
                    "snippet": "A test snippet.",
                    "pagemap": {
                        "cse_image": [{"src": "https://example.com/image.jpg"}],
                        "cse_thumbnail": [
                            {
                                "src": "https://example.com/thumb.jpg",
                                "width": "100",
                                "height": "100",
                            }
                        ],
                    },
                },
                {
                    "title": "No Image Title",
                    "link": "https://example.com/no-image",
                    "snippet": "No image here.",
                },
            ]
        }
        search = GoogleSearchAPIWrapper(  # type: ignore[call-arg]
            google_api_key="key", google_cse_id="cse"
        )
        output = search.results("test", num_results=2)
        assert len(output) == 2
        assert output[0]["image"] == "https://example.com/image.jpg"
        assert output[0]["thumbnail"] == "https://example.com/thumb.jpg"
        assert "image" not in output[1]
        assert "thumbnail" not in output[1]
