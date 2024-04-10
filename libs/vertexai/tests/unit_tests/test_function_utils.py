from enum import Enum
from typing import Optional, Sequence

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from langchain_google_vertexai.functions_utils import (
    _format_tool_to_vertex_function,
    _get_parameters_from_schema,
)


def test_format_tool_to_vertex_function():
    @tool
    def get_datetime() -> str:
        """Gets the current datetime"""
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%d")

    schema = _format_tool_to_vertex_function(get_datetime)  # type: ignore

    assert schema["name"] == "get_datetime"
    assert schema["description"] == "get_datetime() -> str - Gets the current datetime"
    assert "parameters" in schema
    assert "required" not in schema["parameters"]

    @tool
    def sum_two_numbers(a: float, b: float) -> str:
        """Sum two numbers 'a' and 'b'.

        Returns:
            a + b in string format
        """
        return str(a + b)

    schema = _format_tool_to_vertex_function(sum_two_numbers)  # type: ignore

    assert schema["name"] == "sum_two_numbers"
    assert "parameters" in schema
    assert len(schema["parameters"]["required"]) == 2

    @tool
    def do_something_optional(a: float, b: float = 0) -> str:
        """Some description"""
        return str(a + b)

    schema = _format_tool_to_vertex_function(do_something_optional)  # type: ignore

    assert schema["name"] == "do_something_optional"
    assert "parameters" in schema
    assert len(schema["parameters"]["required"]) == 1


def test_get_parameters_from_schema():
    class StringEnum(str, Enum):
        pear = "pear"
        banana = "banana"

    class A(BaseModel):
        """Class A"""

        int_field: Optional[int]

    class B(BaseModel):
        object_field: Optional[A]
        array_field: Sequence[A]
        int_field: int = Field(description="int field", min=0, max=10)
        str_field: str = Field(
            min_length=1, max_length=10, pattern="^[A-Z]{1,10}$", example="ABCD"
        )
        str_enum_field: StringEnum

    schema = B.schema()
    result = _get_parameters_from_schema(schema)

    assert result == {
        "properties": {
            "object_field": {
                "properties": {"int_field": {"type": "integer", "title": "Int Field"}},
                "type": "object",
                "description": "Class A",
                "title": "A",
            },
            "array_field": {
                "items": {
                    "properties": {
                        "int_field": {"type": "integer", "title": "Int Field"}
                    },
                    "type": "object",
                    "description": "Class A",
                    "title": "A",
                },
                "type": "array",
                "title": "Array Field",
            },
            "int_field": {
                "max": 10,
                "type": "integer",
                "min": 0,
                "description": "int field",
                "title": "Int Field",
            },
            "str_field": {
                "minLength": 1,
                "type": "string",
                "maxLength": 10,
                "example": "ABCD",
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
            },
            "str_enum_field": {
                "type": "string",
                "description": "An enumeration.",
                "title": "StringEnum",
                "enum": ["pear", "banana"],
            },
        },
        "type": "object",
        "title": "B",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
