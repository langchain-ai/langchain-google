from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


def validate_column_in_bq_schema(
    columns, column_name: str, expected_types: list, expected_modes: list
):
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


def cast_proto_type(column: str, value: Any):
    if column.startswith("int"):
        return int(value)
    elif column.startswith("double"):
        return float(value)
    elif column.startswith("bool"):
        return bool(value)
    return value


class EnvConfig(BaseModel):
    bq_client: Optional[Any] = None
    project_id: Optional[str] = None
    dataset_name: Optional[str] = None
    table_name: Optional[str] = None
    location: Optional[str] = None
    extra_fields: Union[Dict[str, str], None] = None
    table_schema: Optional[dict] = None
    content_field: Optional[str] = "content"
    text_embedding_field: Optional[str] = "text_embedding"
    doc_id_field: Optional[str] = "doc_id"
    embedding_dimension: Optional[int] = None
    full_table_id: Optional[str] = None
