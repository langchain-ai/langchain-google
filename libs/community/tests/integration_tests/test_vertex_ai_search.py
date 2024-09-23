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

from google.cloud.discoveryengine_v1beta.types import SearchResponse
from google.cloud.discoveryengine_v1beta import Document as DiscoveryEngineDocument
from langchain_google_community.vertex_ai_search import VertexAISearchRetriever

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
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
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
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
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
    retriever = VertexAISearchRetriever(data_store_id=data_store_id)
    documents = retriever.get_relevant_documents("What are Alphabet's Other Bets?")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content
        assert doc.metadata["id"]
        assert doc.metadata["source"]


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

@pytest.mark.parametrize("engine_data_type, get_extractive_answers, segment_config, expected_spec", [
    (0, True, None, {
        "max_extractive_answer_count": 1
    }),
    (0, False, {"num_previous_segments": 1, "num_next_segments": 1}, {
        "max_extractive_segment_count": 1,
        "num_previous_segments": 1,
        "num_next_segments": 1,
        "return_extractive_segment_score": False
    }),
    (0, False, {"num_previous_segments": 2, "num_next_segments": 3}, {
        "max_extractive_segment_count": 1,
        "num_previous_segments": 2,
        "num_next_segments": 3,
        "return_extractive_segment_score": False
    }),
    (0, False, {"num_previous_segments": 3, "num_next_segments": 2}, {
        "max_extractive_segment_count": 1,
        "num_previous_segments": 3,
        "num_next_segments": 2,
        "return_extractive_segment_score": False
    }),
    (0, False, {"num_previous_segments": 3, "num_next_segments": 2, "return_extractive_segment_score": True}, {
        "max_extractive_segment_count": 1,
        "num_previous_segments": 3,
        "num_next_segments": 2,
        "return_extractive_segment_score": True
    }),
])
@pytest.mark.extended
def test_get_content_spec_kwargs(engine_data_type, get_extractive_answers, segment_config, expected_spec):
    retriever_params = {
        "project_id": "mock-project",
        "data_store_id": "mock-data-store",
        "location_id": "global",
        "engine_data_type": engine_data_type,
        "get_extractive_answers": get_extractive_answers,
    }

    if segment_config:
        retriever_params.update(segment_config)

    retriever = VertexAISearchRetriever(**retriever_params)
    content_spec_kwargs = retriever._get_content_spec_kwargs()
    
    assert content_spec_kwargs is not None
    assert "extractive_content_spec" in content_spec_kwargs
    extractive_content_spec = content_spec_kwargs["extractive_content_spec"]
    
    for key, value in expected_spec.items():
        assert hasattr(extractive_content_spec, key)
        assert getattr(extractive_content_spec, key) == value

@pytest.fixture
def mock_search_response():
    return SearchResponse(
        results=[
            SearchResponse.SearchResult(
                id='mock-id-1',
                document=DiscoveryEngineDocument(
                    name='mock-name-1',
                    id='mock-id-1',
                    struct_data={
                        'url': 'mock-url-1',
                        'title': 'Mock Title 1'
                    },
                    derived_struct_data={
                        'link': 'mock-link-1',
                        'extractive_segments': [
                            {
                                'relevanceScore': 0.9,
                                'previous_segments': [
                                    {'content': 'Mock previous segment 1'},
                                    {'content': 'Mock previous segment 2'},
                                    {'content': 'Mock previous segment 3'}
                                ],
                                'next_segments': [
                                    {'content': 'Mock next segment 1'},
                                    {'content': 'Mock next segment 2'},
                                    {'content': 'Mock next segment 3'}
                                ],
                                'content': 'Mock content 1'
                            }
                        ],
                        'extractive_answers': [
                            {'content': 'Mock extractive answer 1'}
                        ]
                    }
                )
            ),
            SearchResponse.SearchResult(
                id='mock-id-2',
                document=DiscoveryEngineDocument(
                    name='mock-name-2',
                    id='mock-id-2',
                    struct_data={
                        'url': 'mock-url-2',
                        'title': 'Mock Title 2'
                    },
                    derived_struct_data={
                        'link': 'mock-link-2',
                        'extractive_segments': [
                            {
                                'relevanceScore': 0.95,
                                'content': 'Mock content 2'
                            }
                        ],
                        'extractive_answers': [
                            {'content': 'Mock extractive answer 2'}
                        ]
                    }
                )
            )
        ]
    )
@pytest.mark.extended
def test_convert_unstructured_search_response_extractive_segments(mock_search_response):
    retriever = VertexAISearchRetriever(
        project_id="mock-project",
        data_store_id="mock-data-store",
        engine_data_type=0,
        get_extractive_answers=False,
        return_extractive_segment_score=True
    )
    
    documents = retriever._convert_unstructured_search_response(mock_search_response.results, "extractive_segments")
    
    assert len(documents) == 2
    
    # Check first document
    assert documents[0].page_content == "Mock content 1"
    assert documents[0].metadata["id"] == "mock-id-1"
    assert documents[0].metadata["source"] == "mock-link-1"
    assert documents[0].metadata["score"] == 0.9
    assert len(documents[0].metadata["previous_segments"]) == 3
    assert len(documents[0].metadata["next_segments"]) == 3
    
    # Check second document
    assert documents[1].page_content == "Mock content 2"
    assert documents[1].metadata["id"] == "mock-id-2"
    assert documents[1].metadata["source"] == "mock-link-2"
    assert documents[1].metadata["score"] == 0.95
    assert len(documents[1].metadata["previous_segments"]) == 0
    assert len(documents[1].metadata["next_segments"]) == 0

@pytest.mark.extended
def test_convert_unstructured_search_response_extractive_answers(mock_search_response):
    retriever = VertexAISearchRetriever(
        project_id="mock-project",
        data_store_id="mock-data-store",
        engine_data_type=0,
        get_extractive_answers=True
    )
    
    documents = retriever._convert_unstructured_search_response(mock_search_response.results, "extractive_answers")
    
    assert len(documents) == 2
    
    # Check first document
    assert documents[0].page_content == "Mock extractive answer 1"
    assert documents[0].metadata["id"] == "mock-id-1"
    assert documents[0].metadata["source"] == "mock-link-1"
    assert "score" not in documents[0].metadata
    assert "previous_segments" not in documents[0].metadata
    assert "next_segments" not in documents[0].metadata

    # Check second document
    assert documents[1].page_content == "Mock extractive answer 2"
    assert documents[1].metadata["id"] == "mock-id-2"
    assert documents[1].metadata["source"] == "mock-link-2"
    assert "score" not in documents[1].metadata
    assert "previous_segments" not in documents[1].metadata
    assert "next_segments" not in documents[1].metadata
