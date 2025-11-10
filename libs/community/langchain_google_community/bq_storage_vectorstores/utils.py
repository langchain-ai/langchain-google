from typing import Any, Dict

from google.cloud.exceptions import NotFound


def validate_column_in_bq_schema(
    columns: dict, column_name: str, expected_types: list, expected_modes: list
) -> None:
    """Validates a column within a BigQuery schema.

    Args:
        columns: A dictionary of BigQuery SchemaField objects representing
            the table schema.
        column_name: The name of the column to validate.
        expected_types: A list of acceptable data types for the column.
        expected_modes: A list of acceptable modes for the column.

    Raises:
        ValueError: If the column doesn't exist, has an unacceptable type,
            or has an unacceptable mode.
    """

    if column_name not in columns:
        raise ValueError(f"Column {column_name} is missing from the schema.")

    column = columns[column_name]

    if column.field_type not in expected_types:
        raise ValueError(
            f"Column {column_name} must be one of the following types: {expected_types}"
        )

    if column.mode not in expected_modes:
        raise ValueError(
            f"Column {column_name} must be one of the following modes: {expected_modes}"
        )


def doc_match_filter(document: Dict[str, Any], filter: Dict[str, Any]) -> bool:
    for column, value in filter.items():
        # ignore fields that are not part of the document
        if document.get(column, value) != value:
            return False
    return True


def cast_proto_type(column: str, value: Any) -> Any:
    if column.startswith("int"):
        return int(value)
    elif column.startswith("double"):
        return float(value)
    elif column.startswith("bool"):
        return bool(value)
    return value


def check_bq_dataset_exists(client: Any, dataset_id: str) -> bool:
    from google.cloud import bigquery  # type: ignore[attr-defined]

    if not isinstance(client, bigquery.Client):
        raise TypeError("client must be an instance of bigquery.Client")

    try:
        client.get_dataset(dataset_id)  # Make an API request.
        return True
    except NotFound:
        return False
