"""Regression tests for BigQuery vector store filters."""

from typing import Any, Union
from unittest.mock import MagicMock

import pytest
from google.cloud import bigquery

from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryVectorStore,
)


def _store() -> BigQueryVectorStore:
    store = BigQueryVectorStore.model_construct()
    store.table_schema = {"tenant_id": "STRING", "priority": "INTEGER"}
    store._bq_client = MagicMock()
    store.project_id = "project"
    store.dataset_name = "dataset"
    store.table_name = "vectors"
    return store


def test_filter_values_bound_in_get_documents() -> None:
    store = _store()
    payload = "legit' OR tenant_id = 'victim'; DROP TABLE x --"
    store.get_documents(ids=["one"], filter={"tenant_id": payload})
    query, kwargs = store._bq_client.query.call_args
    assert payload not in query[0]
    assert "`tenant_id` = @filter_0" in query[0]
    parameters = kwargs["job_config"].query_parameters
    assert isinstance(parameters[0], bigquery.ArrayQueryParameter)
    assert parameters[1].value == payload


def test_filter_values_bound_in_search() -> None:
    store = _store()
    store._search_embeddings([[0.1]], filter={"priority": 1})
    query, kwargs = store._bq_client.query.call_args
    assert "`priority` = @filter_0" in query[0]
    assert [parameter.name for parameter in kwargs["job_config"].query_parameters] == [
        "filter_0",
        "emb_0",
    ]


@pytest.mark.parametrize(
    "filter_value",
    ["1=1", {"priority": "1 OR 1=1 --"}, {"tenant_id`: OR TRUE --": "x"}],
)
def test_unsafe_filters_rejected(filter_value: Union[dict[str, Any], str]) -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.get_documents(filter=filter_value)
    store._bq_client.query.assert_not_called()
