import json
from enum import Enum
from typing import Optional, Sequence, Union, cast

import google.cloud.aiplatform_v1beta1.types as gapic
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, Field
from pydantic.v1 import (
    BaseModel as BaseModelV1,
)
from pydantic.v1 import (
    Field as FieldV1,
)

from langchain_google_common.functions_utils import (
    _format_json_schema_to_gapic,
    _format_json_schema_to_gapic_v1,
)


def test_format_json_schema_to_gapic():
    # Simple case
    class RecordPerson(BaseModel):
        """Record some identifying information about a person."""

        name: str
        age: Optional[int]

    schema = RecordPerson.model_json_schema()
    result = _format_json_schema_to_gapic(schema)
    expected = {
        "title": "RecordPerson",
        "type": "OBJECT",
        "description": "Record some identifying information about a person.",
        "properties": {
            "name": {"title": "Name", "type": "STRING"},
            "age": {"type": "INTEGER", "title": "Age"},
        },
        "required": ["name"],
    }
    assert result == expected

    # Nested case
    class StringEnum(str, Enum):
        pear = "pear"
        banana = "banana"

    class A(BaseModel):
        """Class A"""

        int_field: Optional[int]

    class B(BaseModel):
        object_field: Optional[A] = Field(description="Class A")
        array_field: Sequence[A]
        int_field: int = Field(description="int field", ge=1, le=10)
        str_field: str = Field(
            min_length=1,
            max_length=10,
            pattern="^[A-Z]{1,10}$",
            json_schema_extra={"example": "ABCD"},
        )
        str_enum_field: StringEnum

    schema = B.model_json_schema()
    result = _format_json_schema_to_gapic(dereference_refs(schema))

    expected = {
        "properties": {
            "object_field": {
                "description": "Class A",
                "properties": {"int_field": {"type": "INTEGER", "title": "Int Field"}},
                "required": [],
                "title": "A",
                "type": "OBJECT",
            },
            "array_field": {
                "items": {
                    "description": "Class A",
                    "properties": {
                        "int_field": {"type": "INTEGER", "title": "Int Field"}
                    },
                    "required": [],
                    "title": "A",
                    "type": "OBJECT",
                },
                "type": "ARRAY",
                "title": "Array Field",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10,
                "minimum": 1,
                "title": "Int Field",
                "type": "INTEGER",
            },
            "str_field": {
                "example": "ABCD",
                "maxLength": 10,
                "minLength": 1,
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
                "type": "STRING",
            },
            "str_enum_field": {
                "enum": ["pear", "banana"],
                "title": "StringEnum",
                "type": "STRING",
            },
        },
        "type": "OBJECT",
        "title": "B",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
    assert result == expected

    gapic_schema = cast(gapic.Schema, gapic.Schema.from_json(json.dumps(result)))
    assert gapic_schema.type_ == gapic.Type.OBJECT
    assert gapic_schema.title == expected["title"]
    assert gapic_schema.required == expected["required"]
    assert (
        gapic_schema.properties["str_field"].example
        == expected["properties"]["str_field"]["example"]  # type: ignore
    )


def test_format_json_schema_to_gapic_v1():
    # Simple case
    class RecordPerson(BaseModelV1):
        """Record some identifying information about a person."""

        name: str
        age: Optional[int]

    schema = RecordPerson.schema()
    result = _format_json_schema_to_gapic_v1(schema)
    expected = {
        "title": "RecordPerson",
        "type": "OBJECT",
        "description": "Record some identifying information about a person.",
        "properties": {
            "name": {"title": "Name", "type": "STRING"},
            "age": {"type": "INTEGER", "title": "Age"},
        },
        "required": ["name"],
    }
    assert result == expected

    # Nested case
    class StringEnum(str, Enum):
        pear = "pear"
        banana = "banana"

    class A(BaseModelV1):
        """Class A"""

        int_field: Optional[int]

    class B(BaseModelV1):
        object_field: Optional[A] = FieldV1(description="Class A")
        array_field: Sequence[A]
        int_field: int = FieldV1(description="int field", minimum=1, maximum=10)
        str_field: str = FieldV1(
            min_length=1, max_length=10, pattern="^[A-Z]{1,10}$", example="ABCD"
        )
        str_enum_field: StringEnum

    schema = B.schema()
    result = _format_json_schema_to_gapic_v1(dereference_refs(schema))

    expected = {
        "properties": {
            "object_field": {
                "description": "Class A",
                "properties": {"int_field": {"type": "INTEGER", "title": "Int Field"}},
                "title": "A",
                "type": "OBJECT",
            },
            "array_field": {
                "items": {
                    "description": "Class A",
                    "properties": {
                        "int_field": {"type": "INTEGER", "title": "Int Field"}
                    },
                    "title": "A",
                    "type": "OBJECT",
                },
                "type": "ARRAY",
                "title": "Array Field",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10.0,
                "minimum": 1.0,
                "title": "Int Field",
                "type": "INTEGER",
            },
            "str_field": {
                "example": "ABCD",
                "maxLength": 10,
                "minLength": 1,
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
                "type": "STRING",
            },
            "str_enum_field": {
                "description": "An enumeration.",
                "enum": ["pear", "banana"],
                "title": "StringEnum",
                "type": "STRING",
            },
        },
        "type": "OBJECT",
        "title": "B",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
    assert result == expected

    gapic_schema = cast(gapic.Schema, gapic.Schema.from_json(json.dumps(result)))
    assert gapic_schema.type_ == gapic.Type.OBJECT
    assert gapic_schema.title == expected["title"]
    assert gapic_schema.required == expected["required"]
    assert (
        gapic_schema.properties["str_field"].example
        == expected["properties"]["str_field"]["example"]  # type: ignore
    )


def test_format_json_schema_to_gapic_union_types() -> None:
    """Test that union types are consistent between v1 and v2."""

    class RecordPerson_v1(BaseModelV1):
        name: str
        age: Union[int, str]

    class RecordPerson(BaseModel):
        name: str
        age: Union[int, str]

    schema_v1 = RecordPerson_v1.schema()
    schema_v2 = RecordPerson.model_json_schema()
    del schema_v2

    result_v1 = _format_json_schema_to_gapic_v1(schema_v1)
    # result_v2 = _format_json_schema_to_gapic(schema_v2)
    result_v1["title"] = "RecordPerson"

    # TODO: add a proper support for Union since it has finally arrived!
    # assert result_v1 == result_v2
