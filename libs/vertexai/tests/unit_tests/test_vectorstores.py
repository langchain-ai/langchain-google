from typing import Dict, List, Union
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_google_vertexai.vectorstores._utils import to_data_points
from langchain_google_vertexai.vectorstores.vectorstores import _BaseVertexAIVectorStore


def test_to_data_points():
    ids = ["Id1"]
    embeddings = [[0.0, 0.0]]
    sparse_embeddings: List[Dict[str, Union[List[int], List[float]]]] = [
        {"values": [0.9, 0.3], "dimensions": [3, 20]}
    ]
    metadatas = [
        {
            "some_string": "string",
            "some_number": 1.1,
            "some_list": ["a", "b"],
            "some_random_object": {"foo": 1, "bar": 2},
        }
    ]

    with pytest.warns():
        result = to_data_points(
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            metadatas=metadatas,
        )

    assert isinstance(result, list)
    assert len(result) == 1

    datapoint = result[0]
    datapoint.datapoint_id == "Id1"
    for component_emb, component_fv in (datapoint.feature_vector, embeddings[0]):
        assert component_emb == pytest.approx(component_fv)

    metadata = metadatas[0]

    restriction_lookup = {
        restriction.namespace: restriction for restriction in datapoint.restricts
    }

    restriction = restriction_lookup.pop("some_string")
    assert restriction.allow_list == [metadata["some_string"]]

    restriction = restriction_lookup.pop("some_list")
    assert restriction.allow_list == metadata["some_list"]

    assert len(restriction_lookup) == 0

    num_restriction_lookup = {
        restriction.namespace: restriction
        for restriction in datapoint.numeric_restricts
    }
    restriction = num_restriction_lookup.pop("some_number")
    assert round(restriction.value_float, 1) == pytest.approx(metadata["some_number"])
    assert len(num_restriction_lookup) == 0


def test_add_texts_with_custom_ids(mocker):
    ids = ["Id1", "Id2"]
    texts = ["Text1", "Text2"]

    VectorStore = object.__new__(_BaseVertexAIVectorStore)
    VectorStore._document_storage = MagicMock()
    VectorStore._embeddings = MagicMock()
    VectorStore._searcher = MagicMock()

    mocker.patch.object(VectorStore, "_generate_unique_ids")

    returned_ids = VectorStore.add_texts(texts=texts, ids=ids)

    assert returned_ids == ids

    VectorStore._generate_unique_ids.assert_not_called()  # type: ignore[attr-defined]
    VectorStore._document_storage.mset.assert_called_once()
    VectorStore._embeddings.embed_documents.assert_called_once()
    VectorStore._searcher.add_to_index.assert_called_once()

    with pytest.raises(ValueError):
        VectorStore.add_texts(texts=texts, ids=["Id1"])
    with pytest.raises(ValueError):
        VectorStore.add_texts(texts=texts, ids=["Id1", "Id2", "Id2"])


def test_add_texts_with_single_string():
    """Test that a single string input is properly handled as one document."""
    single_string = "This is a single string"

    VectorStore = object.__new__(_BaseVertexAIVectorStore)
    VectorStore._document_storage = MagicMock()
    VectorStore._embeddings = MagicMock()
    VectorStore._searcher = MagicMock()

    VectorStore.add_texts(texts=single_string)

    VectorStore._embeddings.embed_documents.assert_called_once_with([single_string])


def test_add_texts_with_embeddings():
    texts = ["Text1", "Text2"]
    embeddings = [[0.1, 0.2, 1.0], [1.0, 0.0, 1.0]]
    embeddings_wrong_length = [[0.0, 0.0, 1.0]]

    VectorStore = object.__new__(_BaseVertexAIVectorStore)
    VectorStore._document_storage = MagicMock()
    VectorStore._embeddings = MagicMock()
    VectorStore._searcher = MagicMock()

    returned_ids = VectorStore.add_texts_with_embeddings(
        texts=texts, embeddings=embeddings
    )

    assert len(returned_ids) == len(texts)

    with pytest.raises(ValueError):
        VectorStore.add_texts_with_embeddings(
            texts=texts, embeddings=embeddings_wrong_length
        )


def test_similarity_search_by_vector_with_score_output_shape():
    embedding = [0.0, 0.5, 0.8]
    sparse_embedding: Dict[str, Union[List[int], List[float]]] = {
        "values": [0.9, 0.3],
        "dimensions": [3, 20],
    }

    VectorStore = object.__new__(_BaseVertexAIVectorStore)
    VectorStore._document_storage = MagicMock()
    VectorStore._embeddings = MagicMock()
    VectorStore._searcher = MagicMock()

    # Mock the searcher to return some sample neighbors
    sample_neighbors = [{"doc_id": "doc1", "dense_score": 0.8, "sparse_score": 0.5}]
    VectorStore._searcher.find_neighbors.return_value = [sample_neighbors]

    # Mock the document storage to return a document for the doc_id
    sample_document = Document(page_content="test content")
    VectorStore._document_storage.mget.return_value = [sample_document]

    # Test with sparse embedding
    result_with_sparse = VectorStore.similarity_search_by_vector_with_score(
        embedding=embedding, sparse_embedding=sparse_embedding
    )

    # Should return tuples with (document, dict_of_scores)
    assert len(result_with_sparse) == 1
    assert len(result_with_sparse[0]) == 2
    assert isinstance(result_with_sparse[0][0], Document)
    assert isinstance(result_with_sparse[0][1], dict)
    assert "dense_score" in result_with_sparse[0][1]
    assert "sparse_score" in result_with_sparse[0][1]
    assert isinstance(result_with_sparse[0][1]["dense_score"], float)
    assert isinstance(result_with_sparse[0][1]["sparse_score"], float)

    # Test without sparse embedding (dense-only search)
    result_without_sparse = VectorStore.similarity_search_by_vector_with_score(
        embedding=embedding
    )

    # Should return tuples with (document, float)
    assert len(result_without_sparse) == 1
    assert len(result_without_sparse[0]) == 2
    assert isinstance(result_without_sparse[0][0], Document)
    assert isinstance(result_without_sparse[0][1], float)
