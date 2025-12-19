from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_google_vertexai.vectorstores._utils import to_data_points
from langchain_google_vertexai.vectorstores.vectorstores import _BaseVertexAIVectorStore


def test_to_data_points() -> None:
    ids = ["Id1"]
    embeddings = [[0.0, 0.0]]
    sparse_embeddings: list[dict[str, list[int] | list[float]]] = [
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


def test_to_data_points_with_integer_metadata() -> None:
    """Test that integer metadata values are properly handled with value_int field."""
    ids = ["Id1"]
    embeddings = [[0.1, 0.2, 0.3]]
    metadatas = [
        {
            "integer_field": 42,
            "float_field": 3.14,
            "string_field": "test",
            "mixed_integers": 100,
        }
    ]

    result = to_data_points(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    assert isinstance(result, list)
    assert len(result) == 1

    datapoint = result[0]
    assert datapoint.datapoint_id == "Id1"

    # Check that we have the expected number of numeric restrictions
    assert (
        len(datapoint.numeric_restricts) == 3
    )  # integer_field, float_field, mixed_integers

    # Create lookup for numeric restrictions
    num_restriction_lookup = {
        restriction.namespace: restriction
        for restriction in datapoint.numeric_restricts
    }

    # Test integer field uses value_int
    integer_restriction = num_restriction_lookup["integer_field"]
    assert hasattr(integer_restriction, "value_int")
    assert integer_restriction.value_int == 42
    # Ensure value_float is 0.0 for integer values (not used)
    assert integer_restriction.value_float == 0.0

    # Test float field uses value_float
    float_restriction = num_restriction_lookup["float_field"]
    assert hasattr(float_restriction, "value_float")
    assert float_restriction.value_float == pytest.approx(3.14)
    # Ensure value_int is 0 for float values (not used)
    assert float_restriction.value_int == 0

    # Test another integer field uses value_int
    mixed_restriction = num_restriction_lookup["mixed_integers"]
    assert hasattr(mixed_restriction, "value_int")
    assert mixed_restriction.value_int == 100
    assert mixed_restriction.value_float == 0.0

    # Check string restrictions are still handled correctly
    restriction_lookup = {
        restriction.namespace: restriction for restriction in datapoint.restricts
    }
    string_restriction = restriction_lookup["string_field"]
    assert string_restriction.allow_list == ["test"]


def test_add_texts_with_custom_ids(mocker) -> None:
    ids = ["Id1", "Id2"]
    texts = ["Text1", "Text2"]

    vectorstore = object.__new__(_BaseVertexAIVectorStore)
    vectorstore._document_storage = MagicMock()
    vectorstore._embeddings = MagicMock()
    vectorstore._searcher = MagicMock()
    vectorstore._searcher._api_version = "v1"  # Set V1 API version

    mocker.patch.object(vectorstore, "_generate_unique_ids")

    returned_ids = vectorstore.add_texts(texts=texts, ids=ids)

    assert returned_ids == ids

    vectorstore._generate_unique_ids.assert_not_called()  # type: ignore[attr-defined]
    vectorstore._document_storage.mset.assert_called_once()
    vectorstore._embeddings.embed_documents.assert_called_once()
    vectorstore._searcher.add_to_index.assert_called_once()

    with pytest.raises(ValueError):
        vectorstore.add_texts(texts=texts, ids=["Id1"])
    with pytest.raises(ValueError):
        vectorstore.add_texts(texts=texts, ids=["Id1", "Id2", "Id2"])


def test_add_texts_with_single_string() -> None:
    """Test that a single string input is properly handled as one document."""
    single_string = "This is a single string"

    vectorstore = object.__new__(_BaseVertexAIVectorStore)
    vectorstore._document_storage = MagicMock()
    vectorstore._embeddings = MagicMock()
    vectorstore._searcher = MagicMock()

    vectorstore.add_texts(texts=single_string)

    vectorstore._embeddings.embed_documents.assert_called_once_with([single_string])


def test_add_texts_with_embeddings() -> None:
    texts = ["Text1", "Text2"]
    embeddings = [[0.1, 0.2, 1.0], [1.0, 0.0, 1.0]]
    embeddings_wrong_length = [[0.0, 0.0, 1.0]]

    vectorstore = object.__new__(_BaseVertexAIVectorStore)
    vectorstore._document_storage = MagicMock()
    vectorstore._embeddings = MagicMock()
    vectorstore._searcher = MagicMock()

    returned_ids = vectorstore.add_texts_with_embeddings(
        texts=texts, embeddings=embeddings
    )

    assert len(returned_ids) == len(texts)

    with pytest.raises(ValueError):
        vectorstore.add_texts_with_embeddings(
            texts=texts, embeddings=embeddings_wrong_length
        )


def test_similarity_search_by_vector_with_score_output_shape() -> None:
    embedding = [0.0, 0.5, 0.8]
    sparse_embedding: dict[str, list[int] | list[float]] = {
        "values": [0.9, 0.3],
        "dimensions": [3, 20],
    }

    vectorstore = object.__new__(_BaseVertexAIVectorStore)
    vectorstore._document_storage = MagicMock()
    vectorstore._embeddings = MagicMock()
    vectorstore._searcher = MagicMock()
    vectorstore._searcher._api_version = "v1"  # Set V1 API version

    # Mock the searcher to return some sample neighbors
    sample_neighbors = [{"doc_id": "doc1", "dense_score": 0.8, "sparse_score": 0.5}]
    vectorstore._searcher.find_neighbors.return_value = [sample_neighbors]

    # Mock the document storage to return a document for the doc_id
    sample_document = Document(page_content="test content")
    vectorstore._document_storage.mget.return_value = [sample_document]

    # Test with sparse embedding
    result_with_sparse = vectorstore.similarity_search_by_vector_with_score(
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
    result_without_sparse = vectorstore.similarity_search_by_vector_with_score(
        embedding=embedding
    )

    # Should return tuples with (document, float)
    assert len(result_without_sparse) == 1
    assert len(result_without_sparse[0]) == 2
    assert isinstance(result_without_sparse[0][0], Document)
    assert isinstance(result_without_sparse[0][1], float)
