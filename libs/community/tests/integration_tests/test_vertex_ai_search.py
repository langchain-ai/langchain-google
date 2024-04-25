"""Test Google Vertex AI Search retriever.

You need to create a Vertex AI Search app and populate it
with data to run the integration tests.
Follow the instructions in the example notebook:
google_vertex_ai_search.ipynb
to set up the app and configure authentication.

Set the following environment variables before the tests:
export PROJECT_ID=... - set to your Google Cloud project ID
export DATA_STORE_ID=... - the ID of the search engine to use for the test
"""

import os
import pytest
from langchain_core.documents import Document

from langchain_google_community import (
    VertexAIMultiTurnSearchRetriever,
    VertexAISearchRetriever,
    VertexAISearchSummaryTool,
)


def test_google_vertex_ai_search_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    retriever = VertexAIMultiTurnSearchRetriever(data_store_id=google_cse_id)
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


def test_google_vertex_ai_multiturnsearch_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
    retriever = VertexAISearchRetriever(data_store_id=google_cse_id)
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


def test_vertex_search_tool() -> None:
    tool = VertexAISearchSummaryTool(
        name="vertex-search", description="Vertex Search Tool"
    )

    response = tool.run("How many Champion's Leagues has Real Madrid won?")

    assert isinstance(response, str)
