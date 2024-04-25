from unittest.mock import Mock, patch

import pytest
from google.api_core import exceptions as core_exceptions
from google.cloud import discoveryengine_v1alpha
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from pytest import approx

from langchain_google_community.rank.rank import VertexAIRank


# Fixtures for common setup
@pytest.fixture
def mock_rank_service_client():
    mock_client = Mock(spec=discoveryengine_v1alpha.RankServiceClient)
    mock_client.rank.return_value = discoveryengine_v1alpha.RankResponse(
        records=[
            discoveryengine_v1alpha.RankingRecord(
                id="1", content="Document 1", title="Title 1", score=0.9
            ),
            discoveryengine_v1alpha.RankingRecord(
                id="2", content="Document 2", title="Title 2", score=0.8
            ),
        ]
    )
    return mock_client

@pytest.fixture
def ranker(mock_rank_service_client):
    return VertexAIRank(
        project_id="test-project",
        location_id="test-location",
        ranking_config="test-config",
        title_field="source",
        client=mock_rank_service_client,
    )

# Unit tests
def test_vertex_ai_ranker_initialization():
    ranker = VertexAIRank(
        project_id="test-project",
        location_id="test-location",
        ranking_config="test-config",
        title_field="source",
    )
    assert ranker.project_id == "test-project"
    assert ranker.location_id == "test-location"
    assert ranker.ranking_config == "test-config"
    assert ranker.title_field == "source"

@patch("langchain_google_community.rank.rank.discoveryengine_v1alpha.RankServiceClient")
def test_rerank_documents(mock_rank_service_client_class, ranker):
    documents = [
        Document(page_content="Document 1", metadata={"source": "Title 1"}),
        Document(page_content="Document 2", metadata={"source": "Title 2"}),
    ]
    reranked_documents = ranker._rerank_documents(query="test query", documents=documents)
    print(reranked_documents)
    assert len(reranked_documents) == 2
    assert reranked_documents[0].page_content == "Document 1"
    assert reranked_documents[0].metadata["relevance_score"] == approx(0.9)
    assert reranked_documents[0].metadata["source"] == "Title 1"
    assert reranked_documents[1].page_content == "Document 2"
    assert reranked_documents[1].metadata["relevance_score"] == approx(0.8)
    assert reranked_documents[1].metadata["source"] == "Title 2"