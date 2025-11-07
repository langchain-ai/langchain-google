import json
import uuid
import warnings
from typing import Any

from google.cloud.aiplatform import MatchingEngineIndex
from google.cloud.aiplatform.compat.types import (  # type: ignore[attr-defined, unused-ignore]
    matching_engine_index as meidx_types,
)
from google.cloud.storage import Bucket  # type: ignore[import-untyped, unused-ignore]


def stream_update_index(
    index: MatchingEngineIndex, data_points: list["meidx_types.IndexDataPoint"]
) -> None:
    """Updates an index using stream updating.

    Args:
        index: Vector search index.
        data_points: List of `IndexDataPoint`.
    """
    index.upsert_datapoints(data_points)


def batch_update_index(
    index: MatchingEngineIndex,
    data_points: list["meidx_types.IndexDataPoint"],
    *,
    staging_bucket: Bucket,
    prefix: str | None = None,
    file_name: str = "documents.json",
    is_complete_overwrite: bool = False,
) -> None:
    """Updates an index using batch updating.

    Args:
        index: Vector search index.
        data_points: List of `IndexDataPoint`.
        staging_bucket: Bucket where the staging data is stored. Must be in the same
            region as the index.
        prefix: Prefix for the blob name. If not provided an unique iid will be
            generated.
        file_name: File name of the staging embeddings. By default `'documents.json'`.
        is_complete_overwrite: Whether is an append or overwrite operation.
    """
    if prefix is None:
        prefix = str(uuid.uuid4())

    records = data_points_to_batch_update_records(data_points)

    file_content = "\n".join(json.dumps(record) for record in records)

    blob = staging_bucket.blob(f"{prefix}/{file_name}")
    blob.upload_from_string(file_content)

    contents_delta_uri = f"gs://{staging_bucket.name}/{prefix}"

    index.update_embeddings(
        contents_delta_uri=contents_delta_uri,
        is_complete_overwrite=is_complete_overwrite,
    )


def to_data_points(
    ids: list[str],
    embeddings: list[list[float]],
    sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
) -> list["meidx_types.IndexDataPoint"]:
    """Converts triplets id, embedding, metadata into `IndexDataPoints` instances.

    Only metadata with values of type string, numeric or list of string will be
    considered for the filtering.

    Args:
        ids: List of unique IDs.
        embeddings: List of feature representatitons.
        metadatas: List of metadatas.
    """
    if metadatas is None:
        metadatas = [{}] * len(ids)

    if sparse_embeddings is None:
        sparse_embeddings = [{"values": [], "dimensions": []}] * len(ids)

    data_points = []
    ignored_fields = set()

    for id_, embedding, sparse_embedding, metadata in zip(
        ids, embeddings, sparse_embeddings, metadatas, strict=False
    ):
        restricts = []
        numeric_restricts = []

        for namespace, value in metadata.items():
            if not isinstance(namespace, str):
                msg = "All metadata keys must be strings"  # type: ignore[unreachable, unused-ignore]
                raise ValueError(msg)

            if isinstance(value, str):
                restriction = meidx_types.IndexDatapoint.Restriction(
                    namespace=namespace, allow_list=[value]
                )
                restricts.append(restriction)
            elif isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                restriction = meidx_types.IndexDatapoint.Restriction(
                    namespace=namespace, allow_list=value
                )
                restricts.append(restriction)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, int) and not isinstance(value, bool):
                    restriction = meidx_types.IndexDatapoint.NumericRestriction(
                        namespace=namespace, value_int=value
                    )
                elif isinstance(value, float):
                    restriction = meidx_types.IndexDatapoint.NumericRestriction(
                        namespace=namespace, value_float=value
                    )
                numeric_restricts.append(restriction)
            else:
                ignored_fields.add(namespace)

        if len(ignored_fields) > 0:
            warnings.warn(
                f"Some values in fields {', '.join(ignored_fields)} are not usable for"
                f" restrictions. In order to be used they must be str, list[str] or"
                f" numeric."
            )

        data_point = meidx_types.IndexDatapoint(
            datapoint_id=id_,
            feature_vector=embedding,
            sparse_embedding=sparse_embedding,
            restricts=restricts,
            numeric_restricts=numeric_restricts,
        )

        data_points.append(data_point)

    return data_points


def data_points_to_batch_update_records(
    data_points: list["meidx_types.IndexDataPoint"],
) -> list[dict[str, Any]]:
    """Given a list of datapoints, generates a list of records in the input format
    required to do a bactch update.

    Args:
        data_points: List of `IndexDataPoints`.

    Returns:
        List of records with the format needed to do a batch update.
    """
    records = []

    for data_point in data_points:
        record = {
            "id": data_point.datapoint_id,
            "embedding": list(data_point.feature_vector),
            "restricts": [
                {
                    "namespace": restrict.namespace,
                    "allow": list(restrict.allow_list),
                }
                for restrict in data_point.restricts
            ],
            "numeric_restricts": [
                {
                    "namespace": restrict.namespace,
                    **(
                        {"value_int": restrict.value_int}
                        if hasattr(restrict, "value_int")
                        and restrict.value_int is not None
                        else {}
                    ),
                    **(
                        {"value_float": restrict.value_float}
                        if hasattr(restrict, "value_float")
                        and restrict.value_float is not None
                        else {}
                    ),
                }
                for restrict in data_point.numeric_restricts
            ],
        }

        if (
            hasattr(data_point, "sparse_embedding")
            and data_point.sparse_embedding is not None
        ):
            record["sparse_embedding"] = {
                "values": list(data_point.sparse_embedding.values),
                "dimensions": list(data_point.sparse_embedding.dimensions),
            }

        records.append(record)

    return records
