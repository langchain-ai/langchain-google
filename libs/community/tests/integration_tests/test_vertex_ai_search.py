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

import json
import os
import pickle
from typing import Dict, Optional

import cloudpickle
import pytest
from langchain_core.documents import Document
from langchain_core.load import load

from langchain_google_community import (
    VertexAIMultiTurnSearchRetriever,
    VertexAISearchRetriever,
    VertexAISearchSummaryTool,
)

boost_spec = {
    "condition_boost_specs": [
        {
            "condition": "true",
            "boost_control_spec": {
                "field_name": "dateModified",
                "attribute_type": "FRESHNESS",
                "interpolation_type": "LINEAR",
                "control_points": [
                    {"attribute_value": "7d", "boost_amount": 0.9},
                    {"attribute_value": "30d", "boost_amount": 0.7},
                ],
            },
        }
    ]
}


@pytest.mark.extended
@pytest.mark.parametrize("spec", [None, boost_spec])
def test_google_vertex_ai_search_get_relevant_documents(spec: Optional[Dict]) -> None:
    """Test the get_relevant_documents() method."""
    data_store_id = os.environ["DATA_STORE_ID"]
    if spec:
        retriever = VertexAIMultiTurnSearchRetriever(
            data_store_id=data_store_id, boost_spec=spec
        )
    else:
        retriever = VertexAIMultiTurnSearchRetriever(data_store_id=data_store_id)
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


@pytest.mark.extended
def test_google_vertex_ai_search_boostspec() -> None:
    """Test the get_relevant_documents() method."""
    data_store_id = os.environ["DATA_STORE_ID"]
    retriever = VertexAIMultiTurnSearchRetriever(data_store_id=data_store_id)
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


@pytest.mark.extended
def test_google_vertex_ai_multiturnsearch_get_relevant_documents() -> None:
    """Test the get_relevant_documents() method."""
    data_store_id = os.environ["DATA_STORE_ID"]
    retriever = VertexAISearchRetriever(
        data_store_id=data_store_id, get_extractive_answers=True
    )
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


@pytest.mark.extended
def test_google_vertex_ai_multiturnsearch_get_relevant_documents_segments() -> None:
    """Test the get_relevant_documents() method."""
    data_store_id = os.environ["DATA_STORE_ID"]
    retriever = VertexAISearchRetriever(
        data_store_id=data_store_id,
        max_extractive_segment_count=1,
        return_extractive_segment_score=True,
    )
    documents = retriever.invoke("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]
        assert doc.metadata["relevance_score"]
        assert "previous_segments" in doc.metadata
        assert "next_segments" in doc.metadata


@pytest.mark.extended
def test_vertex_search_tool() -> None:
    data_store_id = os.environ["DATA_STORE_ID"]
    tool = VertexAISearchSummaryTool(  # type: ignore[call-arg, call-arg, call-arg]
        name="vertex-search",
        description="Vertex Search Tool",
        data_store_id=data_store_id,
    )

    response = tool.run("How many Champion's Leagues has Real Madrid won?")

    assert isinstance(response, str)


@pytest.mark.extended
@pytest.mark.parametrize("spec", [None, boost_spec])
def test_native_serialization(spec: Optional[Dict]) -> None:
    retriever = VertexAISearchRetriever(
        data_store_id="test-data-store", project_id="test-project", boost_spec=spec
    )
    serialized = json.dumps(retriever.to_json())
    retriever_loaded = load(
        json.loads(serialized), valid_namespaces=["langchain_google_community"]
    )
    assert retriever.model_dump() == retriever_loaded.model_dump()


@pytest.mark.extended
@pytest.mark.parametrize("spec", [None, boost_spec])
def test_cloudpickle(spec: Optional[Dict]) -> None:
    retriever = VertexAISearchRetriever(
        data_store_id="test-data-store", project_id="test-project", boost_spec=spec
    )
    serialized = cloudpickle.dumps(retriever)
    retriever_loaded = pickle.loads(serialized)
    assert retriever.model_dump() == retriever_loaded.model_dump()
