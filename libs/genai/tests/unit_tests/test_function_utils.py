import google.ai.generativelanguage as glm
from langchain_core.tools import tool

from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)


def test_format_tool_to_genai_function() -> None:
    @tool
    def get_datetime() -> str:
        """Gets the current datetime"""
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%d")

    schema = convert_to_genai_function_declarations([get_datetime])
    function_declaration = schema[0].function_declarations[0]
    assert function_declaration.name == "get_datetime"
    assert (
        function_declaration.description
        == "get_datetime() -> str - Gets the current datetime"
    )
    assert function_declaration.parameters
    assert function_declaration.parameters.required == []

    @tool
    def sum_two_numbers(a: float, b: float) -> str:
        """Sum two numbers 'a' and 'b'.

        Returns:
           a + b in string format
        """
        return str(a + b)

    schema = convert_to_genai_function_declarations([sum_two_numbers])  # type: ignore
    function_declaration = schema[0].function_declarations[0]
    assert function_declaration.name == "sum_two_numbers"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 2

    @tool
    def do_something_optional(a: float, b: float = 0) -> str:
        """Some description"""
        return str(a + b)

    schema = convert_to_genai_function_declarations([do_something_optional])  # type: ignore
    function_declaration = schema[0].function_declarations[0]
    assert function_declaration.name == "do_something_optional"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 1


def test_format_tooldict_to_genai_function() -> None:
    calculator = {
        "function_declarations": [
            glm.FunctionDeclaration(
                name="multiply",
                description="Returns the product of two numbers.",
                parameters=glm.Schema(
                    type=glm.Type.OBJECT,
                    properties={
                        "a": glm.Schema(type=glm.Type.NUMBER),
                        "b": glm.Schema(type=glm.Type.NUMBER),
                    },
                    required=["a", "b"],
                ),
            )
        ]
    }
    schema = convert_to_genai_function_declarations(calculator)
    assert schema[0] == calculator


def test_format_native_dict_to_genai_function() -> None:
    calculator = {
        "function_declarations": [
            {
                "name": "multiply",
                "description": "Returns the product of two numbers.",
            }
        ]
    }
    schema = convert_to_genai_function_declarations(calculator)
    assert schema[0] == calculator


def test_format_dict_to_genai_function() -> None:
    calculator = {
        "function_declarations": [
            {
                "name": "search",
                "description": "Returns the product of two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"description": "search query", "type": "string"},
                    },
                },
            }
        ]
    }
    schema = convert_to_genai_function_declarations(calculator)
    function_declaration = schema[0].function_declarations[0]
    assert function_declaration.name == "search"
    assert function_declaration.parameters
    assert function_declaration.parameters.required == []
