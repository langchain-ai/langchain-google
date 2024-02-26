from langchain_core.tools import tool

from langchain_google_vertexai.functions_utils import _format_tool_to_vertex_function


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
