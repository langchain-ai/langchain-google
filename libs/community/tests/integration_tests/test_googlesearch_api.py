"""Integration test for Google Search API Wrapper."""

import os

import pytest

from langchain_google_community.search import GoogleSearchAPIWrapper


@pytest.mark.extended
def test_call() -> None:
    """Test that call gives the correct answer."""
    google_api_key = os.environ["GOOGLE_API_KEY"]
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    search = GoogleSearchAPIWrapper(  # type: ignore[call-arg]
        google_api_key=google_api_key, google_cse_id=google_cse_id
    )
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output


@pytest.mark.extended
def test_result_with_params_call() -> None:
    """Test that call gives the correct answer with extra params."""
    google_api_key = os.environ["GOOGLE_API_KEY"]
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    search = GoogleSearchAPIWrapper(  # type: ignore[call-arg]
        google_api_key=google_api_key, google_cse_id=google_cse_id
    )
    output = search.results(
        query="What was Obama's first name?",
        num_results=5,
        search_params={"cr": "us", "safe": "active"},
    )
    assert len(output)
