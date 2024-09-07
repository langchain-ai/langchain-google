from enum import Enum
from typing import Optional, Sequence

from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1

from langchain_google_vertexai._json_schema_utils import transform_schema_v2_to_v1


def test_transform_schema_v2_to_v1() -> None:
    class Person(BaseModel):
        name: str
        age: Optional[int]

    class PersonV1(BaseModelV1):
        name: str
        age: Optional[int]

    schema = Person.model_json_schema()
    schema_v1 = PersonV1.schema()

    assert schema_v1["title"] == "PersonV1"
    schema_v1["title"] = "Person"

    result = transform_schema_v2_to_v1(schema)
    assert result == schema_v1


def test_transform_schema_v2_to_v1_nested() -> None:
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
            example="ABCD",  # type: ignore[call-arg]
        )
        str_enum_field: StringEnum

    schema = transform_schema_v2_to_v1(dereference_refs(B.model_json_schema()))
    expected = {
        "definitions": {
            "A": {
                "description": "Class A",
                "properties": {"int_field": {"title": "Int Field", "type": "integer"}},
                "required": [],
                "title": "A",
                "type": "object",
            },
            "StringEnum": {
                "enum": ["pear", "banana"],
                "title": "StringEnum",
                "type": "string",
            },
        },
        "properties": {
            "object_field": {
                "description": "Class A",
                "properties": {"int_field": {"title": "Int Field", "type": "integer"}},
                "required": [],
                "title": "A",
                "type": "object",
            },
            "array_field": {
                "items": {
                    "description": "Class A",
                    "properties": {
                        "int_field": {"title": "Int Field", "type": "integer"}
                    },
                    "required": [],
                    "title": "A",
                    "type": "object",
                },
                "title": "Array Field",
                "type": "array",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10,
                "minimum": 1,
                "title": "Int Field",
                "type": "integer",
            },
            "str_field": {
                "example": "ABCD",
                "maxLength": 10,
                "minLength": 1,
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
                "type": "string",
            },
            "str_enum_field": {
                "enum": ["pear", "banana"],
                "title": "StringEnum",
                "type": "string",
            },
        },
        "required": ["array_field", "int_field", "str_enum_field", "str_field"],
        "title": "B",
        "type": "object",
    }
    assert schema == expected
