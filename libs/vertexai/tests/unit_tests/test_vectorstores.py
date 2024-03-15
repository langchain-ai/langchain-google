import pytest

from langchain_google_vertexai.vectorstores._utils import to_data_points


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
