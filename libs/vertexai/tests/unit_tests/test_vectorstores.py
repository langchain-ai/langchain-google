from unittest.mock import MagicMock

import pytest

from langchain_google_vertexai.vectorstores._utils import to_data_points
from langchain_google_vertexai.vectorstores.vectorstores import _BaseVertexAIVectorStore


def test_to_data_points():
    ids = ["Id1"]
    embeddings = [[0.0, 0.0]]
    metadatas = [
        {
            "some_string": "string",
            "some_number": 1.1,
            "some_list": ["a", "b"],
            "some_random_object": {"foo": 1, "bar": 2},
        }
    ]

    with pytest.warns():
        result = to_data_points(ids, embeddings, metadatas)

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
