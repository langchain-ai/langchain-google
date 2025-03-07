from typing import Any, Generator, List, Optional, Union
from unittest.mock import MagicMock, patch

import google.ai.generativelanguage as glm
import pytest
from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from typing_extensions import Annotated

from langchain_google_genai._function_utils import (
    _tool_choice_to_tool_config,
    _ToolConfigDict,
    convert_to_genai_function_declarations,
    tool_to_dict,
)


def test_tool_with_anyof_nullable_param() -> None:
    """
    Example test that checks a string parameter marked as Optional,
    verifying it's recognized as a 'string' & 'nullable'.
    """

    @tool(parse_docstring=True)
    def possibly_none(
        a: Optional[str] = None,
    ) -> str:
        """
        A test function whose argument can be a string or None.

        Args:
          a: Possibly none.
        """
        return "value"

    # Convert to OpenAI, then to GenAI, then to dict
    oai_tool: dict[str, Any] = convert_to_openai_tool(possibly_none)
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(
        function_declarations,
        list,
    ), "Expected a list."

    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict), "Expected a dict."

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict), "Expected a dict."

    properties = parameters.get("properties")
    assert isinstance(properties, dict), "Expected a dict."

    a_property = properties.get("a")
    assert isinstance(a_property, dict), "Expected a dict."

    assert a_property.get("type_") == glm.Type.STRING, "Expected 'a' to be STRING."
    assert a_property.get("nullable") is True, "Expected 'a' to be marked as nullable."


def test_tool_with_array_anyof_nullable_param() -> None:
    """
    Checks an array parameter marked as Optional, verifying it's recognized
    as an 'array' & 'nullable', and that the items are correctly typed.
    """

    @tool(parse_docstring=True)
    def possibly_none_list(
        items: Optional[List[str]] = None,
    ) -> str:
        """
        A test function whose argument can be a list of strings or None.

        Args:
          items: Possibly a list of strings or None.
        """
        return "value"

    # Convert to OpenAI tool
    oai_tool = convert_to_openai_tool(possibly_none_list)

    # Convert to GenAI, then to dict
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list), "Expected a list."

    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict), "Expected a dict."

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict), "Expected a dict."

    properties = parameters.get("properties")
    assert isinstance(properties, dict), "Expected a dict."

    items_property = properties.get("items")
    assert isinstance(items_property, dict), "Expected a dict."

    # Assertions
    assert (
        items_property.get("type_") == glm.Type.ARRAY
    ), "Expected 'items' to be ARRAY."
    assert items_property.get("nullable"), "Expected 'items' to be marked as nullable."
    # Check that the array items are recognized as strings

    items = items_property.get("items")
    assert isinstance(items, dict), "Expected 'items' to be a dict."

    assert items.get("type_") == glm.Type.STRING, "Expected array items to be STRING."


def test_tool_with_nested_object_anyof_nullable_param() -> None:
    """
    Checks an object parameter (dict) marked as Optional, verifying it's recognized
    as an 'object' but defaults to string if there are no real properties,
    and that it is 'nullable'.
    """

    @tool(parse_docstring=True)
    def possibly_none_dict(
        data: Optional[dict] = None,
    ) -> str:
        """
        A test function whose argument can be an object (dict) or None.

        Args:
          data: Possibly a dict or None.
        """
        return "value"

    # Convert to OpenAI, then to GenAI, then to dict
    oai_tool = convert_to_openai_tool(possibly_none_dict)
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list), "Expected a list."

    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict), "Expected a dict."

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict), "Expected a dict."

    properties = parameters.get("properties")
    assert isinstance(properties, dict), "Expected a dict."

    data_property = properties.get("data")
    assert isinstance(data_property, dict), "Expected a dict."

    assert data_property.get("type_") in [
        glm.Type.OBJECT,
        glm.Type.STRING,
    ], "Expected 'data' to be recognized as an OBJECT or fallback to STRING."
    assert (
        data_property.get("nullable") is True
    ), "Expected 'data' to be marked as nullable."


def test_tool_with_enum_anyof_nullable_param() -> None:
    """
    Checks a parameter with an enum, marked as Optional, verifying it's recognized
    as 'string' & 'nullable', and that the 'enum' field is captured.
    """

    @tool(parse_docstring=True)
    def possibly_none_enum(
        status: Optional[str] = None,
    ) -> str:
        """
        A test function whose argument can be an enum string or None.

        Args:
          status: Possibly one of ("active", "inactive", "pending") or None.
        """
        return "value"

    # Convert to OpenAI tool
    oai_tool = convert_to_openai_tool(possibly_none_enum)

    # Manually override the 'enum' for the 'status' property in the parameters
    oai_tool["function"]["parameters"]["properties"]["status"]["enum"] = [
        "active",
        "inactive",
        "pending",
    ]

    # Convert to GenAI, then to dict
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list), "Expected a list."

    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict), "Expected a dict."

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict), "Expected a dict."

    properties = parameters.get("properties")
    assert isinstance(properties, dict), "Expected a dict."

    status_property = properties.get("status")
    assert isinstance(status_property, dict), "Expected a dict."

    # Assertions
    assert (
        status_property.get("type_") == glm.Type.STRING
    ), "Expected 'status' to be STRING."
    assert (
        status_property.get("nullable") is True
    ), "Expected 'status' to be marked as nullable."
    assert status_property.get("enum") == [
        "active",
        "inactive",
        "pending",
    ], "Expected 'status' to have enum values."


def test_format_tool_to_genai_function() -> None:
    @tool
    def get_datetime() -> str:
        """Gets the current datetime"""
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%d")

    schema = convert_to_genai_function_declarations([get_datetime])
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "get_datetime"
    assert function_declaration.description == "Gets the current datetime"
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
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "sum_two_numbers"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 2

    @tool
    def do_something_optional(a: float, b: float = 0) -> str:
        """Some description"""
        return str(a + b)

    schema = convert_to_genai_function_declarations([do_something_optional])  # type: ignore
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "do_something_optional"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 1


def test_tool_with_annotated_optional_args() -> None:
    @tool(parse_docstring=True)
    def split_documents(
        chunk_size: int,
        knowledge_base: Annotated[Union[list[Document], Document], InjectedToolArg],
        chunk_overlap: Optional[int] = None,
        tokenizer_name: Annotated[Optional[str], InjectedToolArg] = "model",
    ) -> list[Document]:
        """
        Tool.

        Args:
          chunk_size: chunk size.
          knowledge_base: knowledge base.
          chunk_overlap: chunk overlap.
          tokenizer_name: tokenizer name.
        """
        return []

    @tool(parse_docstring=True)
    def search_web(
        query: str,
        engine: str = "Google",
        num_results: int = 5,
        truncate_threshold: Optional[int] = None,
    ) -> list[Document]:
        """
        Tool.

        Args:
          query: query.
          engine: engine.
          num_results: number of results.
          truncate_threshold: truncate threshold.
        """
        return []

    tools = [split_documents, search_web]
    # Convert to OpenAI first to mimic what we do in bind_tools.
    oai_tools = [convert_to_openai_tool(t) for t in tools]
    expected = [
        {
            "name": "split_documents",
            "description": "Tool.",
            "parameters": {
                "type_": 6,
                "properties": {
                    "chunk_overlap": {
                        "type_": 3,
                        "description": "chunk overlap.",
                        "format_": "",
                        "nullable": True,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                    "chunk_size": {
                        "type_": 3,
                        "description": "chunk size.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                },
                "required": ["chunk_size"],
                "format_": "",
                "description": "",
                "nullable": False,
                "enum": [],
                "max_items": "0",
                "min_items": "0",
            },
        },
        {
            "name": "search_web",
            "description": "Tool.",
            "parameters": {
                "type_": 6,
                "properties": {
                    "truncate_threshold": {
                        "type_": 3,
                        "description": "truncate threshold.",
                        "format_": "",
                        "nullable": True,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                    "query": {
                        "type_": 1,
                        "description": "query.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                    "engine": {
                        "type_": 1,
                        "description": "engine.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                    "num_results": {
                        "type_": 3,
                        "description": "number of results.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                    },
                },
                "required": ["query"],
                "format_": "",
                "description": "",
                "nullable": False,
                "enum": [],
                "max_items": "0",
                "min_items": "0",
            },
        },
    ]
    actual = tool_to_dict(convert_to_genai_function_declarations(oai_tools))[
        "function_declarations"
    ]
    assert expected == actual


def test_format_native_dict_to_genai_function() -> None:
    calculator = {
        "function_declarations": [
            {
                "name": "multiply",
                "description": "Returns the product of two numbers.",
            }
        ]
    }
    schema = convert_to_genai_function_declarations([calculator])
    expected = glm.Tool(
        function_declarations=[
            glm.FunctionDeclaration(
                name="multiply",
                description="Returns the product of two numbers.",
                parameters=None,
            )
        ]
    )
    assert schema == expected


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
    schema = convert_to_genai_function_declarations([calculator])
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "search"
    assert function_declaration.parameters
    assert function_declaration.parameters.required == []


@pytest.mark.parametrize("choice", (True, "foo", ["foo"], "any"))
def test__tool_choice_to_tool_config(choice: Any) -> None:
    expected = _ToolConfigDict(
        function_calling_config={
            "mode": "ANY",
            "allowed_function_names": ["foo"],
        },
    )
    actual = _tool_choice_to_tool_config(choice, ["foo"])
    assert expected == actual


def test_tool_to_dict_glm_tool() -> None:
    tool = glm.Tool(
        function_declarations=[
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
    )
    tool_dict = tool_to_dict(tool)
    assert tool == convert_to_genai_function_declarations([tool_dict])


@pytest.fixture
def mock_safe_import() -> Generator[MagicMock, None, None]:
    with patch("langchain_google_genai._function_utils.safe_import") as mock:
        yield mock


def test_tool_to_dict_pydantic() -> None:
    class MyModel(BaseModel):
        name: str
        age: int
        likes: list[str]

    gapic_tool = convert_to_genai_function_declarations([MyModel])
    tool_dict = tool_to_dict(gapic_tool)
    assert gapic_tool == convert_to_genai_function_declarations([tool_dict])


def test_tool_to_dict_pydantic_nested() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    class Models(BaseModel):
        models: list[MyModel]

    gapic_tool = convert_to_genai_function_declarations([Models])
    tool_dict = tool_to_dict(gapic_tool)
    assert tool_dict == {
        "function_declarations": [
            {
                "description": "",
                "name": "Models",
                "parameters": {
                    "description": "",
                    "enum": [],
                    "format_": "",
                    "max_items": "0",
                    "min_items": "0",
                    "nullable": False,
                    "properties": {
                        "models": {
                            "description": "",
                            "enum": [],
                            "format_": "",
                            "items": {
                                "description": "MyModel",
                                "enum": [],
                                "format_": "",
                                "max_items": "0",
                                "min_items": "0",
                                "nullable": False,
                                "properties": {
                                    "age": {
                                        "description": "",
                                        "enum": [],
                                        "format_": "",
                                        "max_items": "0",
                                        "min_items": "0",
                                        "nullable": False,
                                        "properties": {},
                                        "required": [],
                                        "type_": 3,
                                    },
                                    "name": {
                                        "description": "",
                                        "enum": [],
                                        "format_": "",
                                        "max_items": "0",
                                        "min_items": "0",
                                        "nullable": False,
                                        "properties": {},
                                        "required": [],
                                        "type_": 1,
                                    },
                                },
                                "required": ["name", "age"],
                                "type_": 6,
                            },
                            "max_items": "0",
                            "min_items": "0",
                            "nullable": False,
                            "properties": {},
                            "required": [],
                            "type_": 5,
                        }
                    },
                    "required": ["models"],
                    "type_": 6,
                },
            }
        ]
    }


def test_tool_to_dict_pydantic_without_import(mock_safe_import: MagicMock) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    mock_safe_import.return_value = False

    gapic_tool = convert_to_genai_function_declarations([MyModel])
    tool_dict = tool_to_dict(gapic_tool)
    assert gapic_tool == convert_to_genai_function_declarations([tool_dict])
