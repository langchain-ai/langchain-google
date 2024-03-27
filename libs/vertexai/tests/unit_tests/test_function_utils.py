from typing import Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from langchain_google_vertexai.functions_utils import (
    _format_tool_to_vertex_function,
    _get_parameters_from_schema,
    _replace_all_ofs,
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
    class A(BaseModel):
        a1: Optional[int]

    class B(BaseModel):
        b1: Optional[A]
        b2: int = Field(description="f2")
        b3: Union[int, str]

    schema = B.schema()
    result = _get_parameters_from_schema(schema)
    assert result["type"] == "object"
    assert "required" in result
    assert len(result["required"]) == 2

    assert "properties" in result
    assert "b1" in result["properties"]
    assert "b2" in result["properties"]
    assert "b3" in result["properties"]

    assert result["properties"]["b1"]["type"] == "object"
    assert "a1" in result["properties"]["b1"]["properties"]
    assert "required" not in result["properties"]["b1"]
    assert len(result["properties"]["b1"]["properties"]) == 1

    assert "anyOf" in result["properties"]["b3"]


def test__replace_all_ofs() -> None:
    schema = {
        "title": "Relationship",
        "type": "object",
        "properties": {
            "target": {
                "title": "Target",
                "description": "The target of the relationship.",
                "allOf": [
                    {
                        "title": "Node",
                        "type": "object",
                        "description": "An entity in a graph.",
                        "properties": {
                            "id": {"title": "Id", "type": "string"},
                        },
                        "required": ["id"],
                    }
                ],
            }
        },
        "required": ["target"],
    }
    expected = {
        "title": "Relationship",
        "type": "object",
        "properties": {
            "target": {
                "title": "Target",
                "type": "object",
                "description": "The target of the relationship. An entity in a graph.",
                "properties": {
                    "id": {"title": "Id", "type": "string"},
                },
                "required": ["id"],
            }
        },
        "required": ["target"],
    }
    actual = _replace_all_ofs(schema)
    assert actual == expected
    assert _replace_all_ofs(expected) == expected
