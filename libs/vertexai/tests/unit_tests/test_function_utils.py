from enum import Enum
from typing import Any, Optional, Sequence, cast

import pytest
from google.cloud.aiplatform_v1beta1.types import (
    FunctionCallingConfig,
    FunctionDeclaration,
)
from google.cloud.aiplatform_v1beta1.types import (
    ToolConfig as GapicToolConfig,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    convert_to_openai_function,
)
from vertexai.generative_models._generative_models import (  # type: ignore[import-untyped]
    ToolConfig,
)

from langchain_google_vertexai.functions_utils import (
    _format_base_tool_to_vertex_function,
    _format_dict_to_function_declaration,
    _format_tool_config,
    _FunctionCallingConfigDict,
    _get_parameters_from_schema,
    _tool_choice_to_tool_config,
    _ToolConfigDict,
)


def test_format_dict_to_function_declaration():
    @tool
    def search(question: str) -> str:
        "Search"
        return question

    func_desc = convert_to_openai_function(search)

    schema = _format_dict_to_function_declaration(cast(FunctionDescription, func_desc))
    expected = FunctionDeclaration(
        name="search",
        description="search(question: str) -> str - Search",
        parameters={
            "type_": "OBJECT",
            "properties": {"question": {"type_": "STRING"}},
            "required": ["question"],
        },
    )

    assert schema == expected


def test_format_tool_to_vertex_function():
    @tool
    def get_datetime() -> str:
        """Gets the current datetime"""
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%d")

    schema = _format_base_tool_to_vertex_function(get_datetime)  # type: ignore

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

    schema = _format_base_tool_to_vertex_function(sum_two_numbers)  # type: ignore

    assert schema["name"] == "sum_two_numbers"
    assert "parameters" in schema
    assert len(schema["parameters"]["required"]) == 2

    @tool
    def do_something_optional(a: float, b: float = 0) -> str:
        """Some description"""
        return str(a + b)

    schema = _format_base_tool_to_vertex_function(do_something_optional)  # type: ignore

    assert schema["name"] == "do_something_optional"
    assert "parameters" in schema
    assert len(schema["parameters"]["required"]) == 1


def test_format_tool_config_invalid():
    with pytest.raises(ValueError):
        _format_tool_config({})  # type: ignore


def test_format_tool_config():
    tool_config = _format_tool_config(
        {
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.ANY,  # type: ignore[typeddict-item]
                "allowed_function_names": ["my_fun"],
            }
        }
    )
    assert tool_config == GapicToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode=FunctionCallingConfig.Mode.ANY, allowed_function_names=["my_fun"]
        )
    )


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


@pytest.mark.parametrize("choice", (True, "foo", ["foo"], "any"))
def test__tool_choice_to_tool_config(choice: Any) -> None:
    expected = _ToolConfigDict(
        function_calling_config=_FunctionCallingConfigDict(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=["foo"],
        ),
    )
    actual = _tool_choice_to_tool_config(choice, ["foo"])
    assert expected == actual
