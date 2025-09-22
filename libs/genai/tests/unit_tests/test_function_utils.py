import datetime
from collections.abc import Generator
from typing import Annotated, Any, Optional, Union
from unittest.mock import MagicMock, patch

import google.ai.generativelanguage as glm
import pytest
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_google_genai._function_utils import (
    _convert_pydantic_to_genai_function,
    _format_base_tool_to_function_declaration,
    _format_dict_to_function_declaration,
    _format_to_gapic_function_declaration,
    _FunctionDeclarationLike,
    _tool_choice_to_tool_config,
    _ToolConfigDict,
    convert_to_genai_function_declarations,
    replace_defs_in_schema,
    tool_to_dict,
)


def test_tool_with_anyof_nullable_param() -> None:
    """Example test.

    Checks a string parameter marked as Optional, verifying it's recognized as a
    'string' & 'nullable'.
    """

    @tool(parse_docstring=True)
    def possibly_none(
        a: Optional[str] = None,
    ) -> str:
        """A test function whose argument can be a string or None.

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
    """Checks an array parameter marked as Optional.

    Verifying it's recognized As an 'array' & 'nullable', and that the items are
    correctly typed.
    """

    @tool(parse_docstring=True)
    def possibly_none_list(
        items: Optional[list[str]] = None,
    ) -> str:
        """A test function whose argument can be a list of strings or None.

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
    assert items_property.get("type_") == glm.Type.ARRAY, (
        "Expected 'items' to be ARRAY."
    )
    assert items_property.get("nullable"), "Expected 'items' to be marked as nullable."
    # Check that the array items are recognized as strings

    items = items_property.get("items")
    assert isinstance(items, dict), "Expected 'items' to be a dict."

    assert items.get("type_") == glm.Type.STRING, "Expected array items to be STRING."


def test_tool_with_nested_object_anyof_nullable_param() -> None:
    """Checks an object parameter (dict) marked as Optional.

    Verifying it's recognized as an 'object' but defaults to string if there are no real
    properties, and that it is 'nullable'.
    """

    @tool(parse_docstring=True)
    def possibly_none_dict(
        data: Optional[dict] = None,
    ) -> str:
        """A test function whose argument can be an object (dict) or None.

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
    assert data_property.get("nullable") is True, (
        "Expected 'data' to be marked as nullable."
    )


def test_tool_with_enum_anyof_nullable_param() -> None:
    """Checks a parameter with an enum, marked as Optional.

    Verifying it's recognized as 'string' & 'nullable', and that the 'enum' field is
    captured.
    """

    @tool(parse_docstring=True)
    def possibly_none_enum(
        status: Optional[str] = None,
    ) -> str:
        """A test function whose argument can be an enum string or None.

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
    assert status_property.get("type_") == glm.Type.STRING, (
        "Expected 'status' to be STRING."
    )
    assert status_property.get("nullable") is True, (
        "Expected 'status' to be marked as nullable."
    )
    assert status_property.get("enum") == [
        "active",
        "inactive",
        "pending",
    ], "Expected 'status' to have enum values."


# reusable test inputs
def search(question: str) -> str:
    """Search tool."""
    return question


search_tool = tool(search)
search_exp = glm.FunctionDeclaration(
    name="search",
    description="Search tool.",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        description="Search tool.",
        properties={"question": glm.Schema(type=glm.Type.STRING)},
        required=["question"],
        title="search",
    ),
)


class SearchBaseTool(BaseTool):
    def _run(self) -> None:
        pass


search_base_tool = SearchBaseTool(name="search", description="Search tool")
search_base_tool_exp = glm.FunctionDeclaration(
    name=search_base_tool.name,
    description=search_base_tool.description,
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            "__arg1": glm.Schema(type=glm.Type.STRING),
        },
        required=["__arg1"],
    ),
)


class SearchModel(BaseModel):
    """Search model."""

    question: str


search_model_schema = SearchModel.model_json_schema()
search_model_dict = {
    "name": search_model_schema["title"],
    "description": search_model_schema["description"],
    "parameters": search_model_schema,
}
search_model_exp = glm.FunctionDeclaration(
    name="SearchModel",
    description="Search model.",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        description="Search model.",
        properties={
            "question": glm.Schema(type=glm.Type.STRING),
        },
        required=["question"],
        title="SearchModel",
    ),
)

search_model_exp_pyd = glm.FunctionDeclaration(
    name="SearchModel",
    description="Search model.",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            "question": glm.Schema(type=glm.Type.STRING),
        },
        required=["question"],
    ),
)

mock_dict = MagicMock(name="mock_dicts", wraps=_format_dict_to_function_declaration)
mock_base_tool = MagicMock(
    name="mock_base_tool", wraps=_format_base_tool_to_function_declaration
)
mock_pydantic = MagicMock(
    name="mock_pydantic", wraps=_convert_pydantic_to_genai_function
)

SRC_EXP_MOCKS_DESC: list[
    tuple[_FunctionDeclarationLike, glm.FunctionDeclaration, list[MagicMock], str]
] = [
    (search, search_exp, [mock_base_tool], "plain function"),
    (search_tool, search_exp, [mock_base_tool], "LC tool"),
    (search_base_tool, search_base_tool_exp, [mock_base_tool], "LC base tool"),
    (SearchModel, search_model_exp_pyd, [mock_pydantic], "Pydantic model"),
    (search_model_dict, search_model_exp, [mock_dict], "dict"),
]


def test_format_tool_to_genai_function() -> None:
    @tool
    def get_datetime() -> str:
        """Gets the current datetime."""
        return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    schema = convert_to_genai_function_declarations([get_datetime])
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "get_datetime"
    assert function_declaration.description == "Gets the current datetime."
    assert function_declaration.parameters
    assert function_declaration.parameters.required == []

    @tool
    def sum_two_numbers(a: float, b: float) -> str:
        """Sum two numbers 'a' and 'b'.

        Returns:
            a + b in string format
        """
        return str(a + b)

    schema = convert_to_genai_function_declarations([sum_two_numbers])
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "sum_two_numbers"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 2

    @tool
    def do_something_optional(a: float, b: float = 0) -> str:
        """Some description."""
        return str(a + b)

    schema = convert_to_genai_function_declarations([do_something_optional])
    function_declaration = schema.function_declarations[0]
    assert function_declaration.name == "do_something_optional"
    assert function_declaration.parameters
    assert len(function_declaration.parameters.required) == 1

    src = [src for src, _, _, _ in SRC_EXP_MOCKS_DESC]
    fds = [fd for _, fd, _, _ in SRC_EXP_MOCKS_DESC]
    expected = glm.Tool(function_declarations=fds)
    result = convert_to_genai_function_declarations(src)
    assert result == expected

    src_2 = glm.Tool(google_search_retrieval={})
    result = convert_to_genai_function_declarations([src_2])
    assert result == src_2

    src_3: dict[str, Any] = {"google_search_retrieval": {}}
    result = convert_to_genai_function_declarations([src_3])
    assert result == src_2

    src_4 = glm.Tool(google_search={})
    result = convert_to_genai_function_declarations([src_4])
    assert result == src_4

    with pytest.raises(ValueError) as exc_info1:
        _ = convert_to_genai_function_declarations(["fake_tool"])  # type: ignore
    assert str(exc_info1.value).startswith("Unsupported tool")

    with pytest.raises(Exception) as exc_info:
        _ = convert_to_genai_function_declarations(
            [
                glm.Tool(google_search_retrieval={}),
                glm.Tool(google_search_retrieval={}),
            ]
        )
    assert str(exc_info.value).startswith("Providing multiple google_search_retrieval")


def test_tool_with_annotated_optional_args() -> None:
    @tool(parse_docstring=True)
    def split_documents(
        chunk_size: int,
        knowledge_base: Annotated[Union[list[Document], Document], InjectedToolArg],
        chunk_overlap: Optional[int] = None,
        tokenizer_name: Annotated[Optional[str], InjectedToolArg] = "model",
    ) -> list[Document]:
        """Tool.

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
        """Tool.

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
                    "chunk_size": {
                        "type_": 3,
                        "description": "chunk size.",
                        "format_": "",
                        "title": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                    "chunk_overlap": {
                        "type_": 3,
                        "description": "chunk overlap.",
                        "nullable": True,
                        "format_": "",
                        "title": "",
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                },
                "required": ["chunk_size"],
                "format_": "",
                "title": "",
                "description": "",
                "nullable": False,
                "enum": [],
                "max_items": "0",
                "min_items": "0",
                "min_properties": "0",
                "max_properties": "0",
                "min_length": "0",
                "max_length": "0",
                "pattern": "",
                "any_of": [],
                "property_ordering": [],
            },
            "behavior": 0,
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
                        "nullable": True,
                        "format_": "",
                        "title": "",
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                    "engine": {
                        "type_": 1,
                        "description": "engine.",
                        "format_": "",
                        "title": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                    "query": {
                        "type_": 1,
                        "description": "query.",
                        "format_": "",
                        "title": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                    "num_results": {
                        "type_": 3,
                        "description": "number of results.",
                        "format_": "",
                        "title": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "required": [],
                        "min_properties": "0",
                        "max_properties": "0",
                        "min_length": "0",
                        "max_length": "0",
                        "pattern": "",
                        "any_of": [],
                        "property_ordering": [],
                    },
                },
                "required": ["query"],
                "format_": "",
                "title": "",
                "description": "",
                "nullable": False,
                "enum": [],
                "max_items": "0",
                "min_items": "0",
                "min_properties": "0",
                "max_properties": "0",
                "min_length": "0",
                "max_length": "0",
                "pattern": "",
                "any_of": [],
                "property_ordering": [],
            },
            "behavior": 0,
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


@pytest.mark.parametrize("choice", [True, "foo", ["foo"], "any"])
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
                "name": "Models",
                "parameters": {
                    "type_": 6,
                    "properties": {
                        "models": {
                            "type_": 5,
                            "items": {
                                "type_": 6,
                                "description": "MyModel",
                                "properties": {
                                    "age": {
                                        "type_": 3,
                                        "format_": "",
                                        "title": "",
                                        "description": "",
                                        "nullable": False,
                                        "enum": [],
                                        "max_items": "0",
                                        "min_items": "0",
                                        "properties": {},
                                        "required": [],
                                        "min_properties": "0",
                                        "max_properties": "0",
                                        "min_length": "0",
                                        "max_length": "0",
                                        "pattern": "",
                                        "any_of": [],
                                        "property_ordering": [],
                                    },
                                    "name": {
                                        "type_": 1,
                                        "format_": "",
                                        "title": "",
                                        "description": "",
                                        "nullable": False,
                                        "enum": [],
                                        "max_items": "0",
                                        "min_items": "0",
                                        "properties": {},
                                        "required": [],
                                        "min_properties": "0",
                                        "max_properties": "0",
                                        "min_length": "0",
                                        "max_length": "0",
                                        "pattern": "",
                                        "any_of": [],
                                        "property_ordering": [],
                                    },
                                },
                                "required": ["name", "age"],
                                "format_": "",
                                "title": "",
                                "nullable": False,
                                "enum": [],
                                "max_items": "0",
                                "min_items": "0",
                                "min_properties": "0",
                                "max_properties": "0",
                                "min_length": "0",
                                "max_length": "0",
                                "pattern": "",
                                "any_of": [],
                                "property_ordering": [],
                            },
                            "format_": "",
                            "title": "",
                            "description": "",
                            "nullable": False,
                            "enum": [],
                            "max_items": "0",
                            "min_items": "0",
                            "properties": {},
                            "required": [],
                            "min_properties": "0",
                            "max_properties": "0",
                            "min_length": "0",
                            "max_length": "0",
                            "pattern": "",
                            "any_of": [],
                            "property_ordering": [],
                        }
                    },
                    "required": ["models"],
                    "format_": "",
                    "title": "",
                    "description": "",
                    "nullable": False,
                    "enum": [],
                    "max_items": "0",
                    "min_items": "0",
                    "min_properties": "0",
                    "max_properties": "0",
                    "min_length": "0",
                    "max_length": "0",
                    "pattern": "",
                    "any_of": [],
                    "property_ordering": [],
                },
                "description": "",
                "behavior": 0,
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


def test_tool_with_doubly_nested_list_param() -> None:
    """Tests a tool parameter with a doubly nested list (list[list[str]]).

    Verifying that the GAPIC schema correctly represents the nested items.
    """

    @tool(parse_docstring=True)
    def process_nested_data(
        matrix: list[list[str]],
    ) -> str:
        """Processes a matrix (list of lists of strings).

        Args:
            matrix: The nested list data.
        """
        return f"Processed {len(matrix)} rows."

    oai_tool = convert_to_openai_tool(process_nested_data)

    genai_tool = convert_to_genai_function_declarations([oai_tool])

    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list)
    assert len(function_declarations) == 1
    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict)

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict)

    properties = parameters.get("properties")
    assert isinstance(properties, dict)

    matrix_property = properties.get("matrix")
    assert isinstance(matrix_property, dict)

    assert matrix_property.get("type_") == glm.Type.ARRAY, (
        "Expected 'matrix' to be ARRAY."
    )

    items_level1 = matrix_property.get("items")
    assert isinstance(items_level1, dict), "Expected first level 'items' to be a dict."
    assert items_level1.get("type_") == glm.Type.ARRAY, (
        "Expected first level items to be ARRAY."
    )

    items_level2 = items_level1.get("items")
    assert isinstance(items_level2, dict), "Expected second level 'items' to be a dict."
    assert items_level2.get("type_") == glm.Type.STRING, (
        "Expected second level items to be STRING."
    )

    assert "description" in matrix_property
    assert "description" in items_level1
    assert "description" in items_level2


def test_schema_no_defs() -> None:
    schema = {"type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_empty_defs() -> None:
    schema = {"$defs": {}, "type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_simple_ref_replacement() -> None:
    schema = {
        "$defs": {"MyDefinition": {"type": "string"}},
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {"property": {"type": "string"}}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_nested_ref_replacement() -> None:
    schema = {
        "$defs": {
            "MyDefinition": {
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/NameDefinition"}},
            },
            "NameDefinition": {"type": "string"},
        },
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {
        "property": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_recursive_error_self_reference() -> None:
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }
        },
        "root": {"$ref": "#/$defs/Node"},
    }
    with pytest.raises(RecursionError):
        _ = replace_defs_in_schema(schema)


def test_tool_with_union_types() -> None:
    """Test union types with tools.

    Tests that validates tools with Union types in function declarations are correctly
    converted to 'anyOf' in the schema.
    """

    class Helper1(BaseModel):
        """Test helper class 1."""

        x: bool = False

    class Helper2(BaseModel):
        """Test helper class 2."""

        y: str = "1"

    class GetWeather(BaseModel):
        """Get weather information."""

        location: str = "New York, USA"
        date: Union[Helper1, Helper2] = Helper1()

    # Convert to OpenAI, then to GenAI, then to dict
    oai_tool = convert_to_openai_tool(GetWeather)
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

    date_property = properties.get("date")
    assert isinstance(date_property, dict), "Expected a dict."

    # Verify that 'any_of' is present in the date property
    assert "any_of" in date_property, "Expected 'date' to have 'any_of' field."

    any_of = date_property.get("any_of")
    assert isinstance(any_of, list), "Expected 'any_of' to be a list."
    assert len(any_of) == 2, "Expected 'any_of' to have 2 options."

    # Check first option (Helper1)
    helper1 = any_of[0]
    assert isinstance(helper1, dict), "Expected first option to be a dict."
    assert "properties" in helper1, "Expected first option to have properties."
    assert "x" in helper1["properties"], "Expected first option to have 'x' property."
    assert helper1["properties"]["x"]["type_"] == glm.Type.BOOLEAN, (
        "Expected 'x' to be BOOLEAN."
    )

    # Check second option (Helper2)
    helper2 = any_of[1]
    assert isinstance(helper2, dict), "Expected second option to be a dict."
    assert "properties" in helper2, "Expected second option to have properties."
    assert "y" in helper2["properties"], "Expected second option to have 'y' property."
    assert helper2["properties"]["y"]["type_"] == glm.Type.STRING, (
        "Expected 'y' to be STRING."
    )


def test_tool_with_union_primitive_types() -> None:
    """Test union primitive types for tools.

    Tests that validates tools with Union types that include primitive types are
    correctly converted to 'anyOf' in the schema.
    """

    class Helper(BaseModel):
        """Test helper class."""

        value: int = 42

    class SearchQuery(BaseModel):
        """Search query model with a union parameter."""

        query: str = "default query"
        filter: Union[str, Helper] = "default filter"

    # Convert to OpenAI, then to GenAI, then to dict
    oai_tool = convert_to_openai_tool(SearchQuery)
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

    filter_property = properties.get("filter")
    assert isinstance(filter_property, dict), "Expected a dict."

    # Verify that 'any_of' is present in the filter property
    assert "any_of" in filter_property, "Expected 'filter' to have 'any_of' field."

    any_of = filter_property.get("any_of")
    assert isinstance(any_of, list), "Expected 'any_of' to be a list."
    assert len(any_of) == 2, "Expected 'any_of' to have 2 options."

    # One option should be a string
    string_option = next(
        (opt for opt in any_of if opt.get("type_") == glm.Type.STRING), None
    )
    assert string_option is not None, "Expected one option to be a STRING."

    # One option should be an object (Helper)
    object_option = next(
        (opt for opt in any_of if opt.get("type_") == glm.Type.OBJECT), None
    )
    assert object_option is not None, "Expected one option to be an OBJECT."
    assert "properties" in object_option, "Expected object option to have properties."
    assert "value" in object_option["properties"], (
        "Expected object option to have 'value' property."
    )
    assert object_option["properties"]["value"]["type_"] == 3, (
        "Expected 'value' to be NUMBER or INTEGER."
    )


def test_tool_with_nested_union_types() -> None:
    """Test nested union types.

    Tests that validates tools with nested Union types are correctly converted to nested
    'anyOf' structures in the schema.
    """

    class Address(BaseModel):
        """Address model."""

        street: str = "123 Main St"
        city: str = "Anytown"

    class Contact(BaseModel):
        """Contact model."""

        email: str = "user@example.com"
        phone: Optional[str] = None

    class Person(BaseModel):
        """Person model with complex nested unions."""

        name: str
        location: Union[str, Address] = "Unknown"
        contacts: list[Union[str, Contact]] = []

    # Convert to OpenAI, then to GenAI, then to dict
    oai_tool = convert_to_openai_tool(Person)
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)

    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list), "Expected a list."

    fn_decl = function_declarations[0]
    parameters = fn_decl.get("parameters")
    properties = parameters.get("properties")

    assert "name" in properties, "Expected 'name' property to exist"
    assert "location" in properties, "Expected 'location' property to exist"

    # Check location property (direct Union)
    location_property = properties.get("location")
    assert "any_of" in location_property, "Expected 'location' to have 'any_of' field."
    location_any_of = location_property.get("any_of")
    assert len(location_any_of) == 2, "Expected 'location.any_of' to have 2 options."

    # One option should be a string
    string_option = next(
        (opt for opt in location_any_of if opt.get("type_") == glm.Type.STRING), None
    )
    assert string_option is not None, "Expected one location option to be a STRING."

    # One option should be an object (Address)
    address_option = next(
        (opt for opt in location_any_of if opt.get("type_") == glm.Type.OBJECT), None
    )
    assert address_option is not None, "Expected one location option to be an OBJECT."
    assert "properties" in address_option, "Expected address option to have properties"
    assert "city" in address_option["properties"], (
        "Expected Address to have 'city' property."
    )


def test_tool_invocation_with_union_types() -> None:
    """Test invocation with union types.

    Tests that validates tools with Union types can be correctly invoked with either
    type from the union.
    """

    class Configuration(BaseModel):
        """Configuration model."""

        settings: dict[str, str] = {}

    @tool
    def configure_service(service_name: str, config: Union[str, Configuration]) -> str:
        """Configure a service with either a configuration string or object.

        Args:
            service_name: The name of the service to configure
            config: Either a config string or a Configuration object
        """
        return f"Configured {service_name}"

    # Convert to OpenAI tool
    oai_tool = convert_to_openai_tool(configure_service)

    # Convert to GenAI
    genai_tool = convert_to_genai_function_declarations([oai_tool])

    # Get function declaration
    function_declaration = genai_tool.function_declarations[0]

    # Check parameters
    parameters = function_declaration.parameters
    assert parameters is not None, "Expected parameters to exist"
    assert hasattr(parameters, "properties"), "Expected parameters to have properties"

    # Check for config property
    config_property = None
    for prop_name, prop in parameters.properties.items():
        if prop_name == "config":
            config_property = prop
            break

    assert config_property is not None, "Expected 'config' property to exist"
    assert hasattr(config_property, "any_of"), "Expected any_of attribute on config"
    assert len(config_property.any_of) == 2, "Expected config.any_of to have 2 options"

    # Check both variants of the Union type
    type_variants = [option.type for option in config_property.any_of]
    assert glm.Type.STRING in type_variants, "Expected STRING to be one of the variants"
    assert glm.Type.OBJECT in type_variants, "Expected OBJECT to be one of the variants"

    # Find the object variant
    object_variant = None
    for option in config_property.any_of:
        if option.type == glm.Type.OBJECT:
            object_variant = option
            break

    assert object_variant is not None, "Expected to find an object variant"
    assert hasattr(object_variant, "properties"), "Expected object to have properties"

    # Check for settings property
    has_settings = False
    for prop_name in object_variant.properties:
        if prop_name == "settings":
            has_settings = True
            break

    assert has_settings, "Expected object variant to have 'settings' property"


def test_tool_field_union_types() -> None:
    """Test field union types.

    Test that validates Field with Union types in Pydantic models are correctly
    converted to 'anyOf' in the schema.
    """

    class Helper1(BaseModel):
        """Helper class 1."""

        x: bool = False

    class Helper2(BaseModel):
        """Helper class 2."""

        y: str = "1"

    class GetWeather(BaseModel):
        """Get weather information for a location."""

        location: str = Field(
            ..., description="The city and country, e.g. New York, USA"
        )
        date: Union[Helper1, Helper2] = Field(description="Test field")

    # Convert to OpenAI tool
    oai_tool = convert_to_openai_tool(GetWeather)

    # Convert to GenAI
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)

    # Get function declaration
    function_declarations = genai_tool_dict.get("function_declarations", [])
    assert len(function_declarations) > 0, "Expected at least one function declaration"
    fn_decl = function_declarations[0]

    # Check the name and description
    assert fn_decl.get("name") == "GetWeather", "Expected name to be 'GetWeather'"  # type: ignore
    assert "Get weather information" in fn_decl.get("description", ""), (  # type: ignore
        "Expected description to include weather information"
    )

    # Check parameters
    parameters = fn_decl.get("parameters", {})  # type: ignore
    properties = parameters.get("properties", {})

    # Check location property
    assert "location" in properties, "Expected location field in properties"
    location_property = properties.get("location", {})
    assert "description" in location_property, (
        "Expected description field in location property"
    )
    assert (
        location_property.get("description")
        == "The city and country, e.g. New York, USA"
    ), "Expected correct description for location"

    # Check date property (the union type)
    assert "date" in properties, "Expected date field in properties"
    date_property = properties.get("date", {})
    assert "any_of" in date_property, "Expected 'date' to have 'any_of' field"

    any_of = date_property.get("any_of", [])
    assert len(any_of) == 2, "Expected 'any_of' to have 2 options"

    # Extract the titles of the models in the anyOf
    model_titles = []
    for option in any_of:
        if "title" in option:
            title = option.get("title")
            if title is not None:
                model_titles.append(title)

    assert "Helper1" in model_titles, "Expected 'Helper1' to be in the anyOf options"
    assert "Helper2" in model_titles, "Expected 'Helper2' to be in the anyOf options"

    # Check that the required fields include both location and date
    assert "required" in parameters, "Expected required field in parameters"
    required_fields = parameters.get("required", [])
    assert "location" in required_fields, "Expected 'location' to be required"
    assert "date" in required_fields, "Expected 'date' to be required"


def test_union_type_schema_validation() -> None:
    """Test that Union types get proper type_ assignment for Gemini compatibility."""

    class Response(BaseModel):
        """Response to user."""

        response: str

    class Plan(BaseModel):
        """Plan to perform."""

        plan: str

    class Act(BaseModel):
        """Action to perform."""

        action: Union[Response, Plan] = Field(description="Action to perform.")

    # Convert to GenAI function declaration
    openai_func = convert_to_openai_function(Act)
    genai_func = _format_to_gapic_function_declaration(openai_func)

    # The action property should have a valid type (not 0) for Gemini compatibility
    action_prop = genai_func.parameters.properties["action"]
    assert action_prop.type_ == glm.Type.OBJECT, (
        f"Union type should have OBJECT type, got {action_prop.type_}"
    )
    assert action_prop.type_ != 0, "Union type should not have type_ = 0"


def test_optional_dict_schema_validation() -> None:
    """Test that Optional types get proper OBJECT type for Gemini compatibility."""

    class RequestsGetToolInput(BaseModel):
        url: str = Field(description="The URL to send the GET request to")
        params: Optional[dict[str, str]] = Field(
            default={}, description="Query parameters for the GET request"
        )
        output_instructions: str = Field(
            description="Instructions on what information to extract from the response"
        )

    # Convert to GenAI function declaration
    openai_func = convert_to_openai_function(RequestsGetToolInput)
    genai_func = _format_to_gapic_function_declaration(openai_func)

    # The params property should have OBJECT type, not STRING
    params_prop = genai_func.parameters.properties["params"]
    assert params_prop.type_ == glm.Type.OBJECT, (
        f"Optional[dict] should have OBJECT type, got {params_prop.type_}"
    )
    assert params_prop.type_ != glm.Type.STRING, (
        "Optional[dict] should not be converted to STRING type"
    )
    assert params_prop.nullable is True, "Optional[dict] should be nullable"
    assert params_prop.description == "Query parameters for the GET request", (
        "Description should be preserved"
    )


def test_tool_field_enum_array() -> None:
    class ToolInfo(BaseModel):
        kind: list[Literal["foo", "bar"]]

    # Convert to OpenAI tool
    oai_tool = convert_to_openai_tool(ToolInfo)

    # Convert to GenAI
    genai_tool = convert_to_genai_function_declarations([oai_tool])
    genai_tool_dict = tool_to_dict(genai_tool)

    # Get function declaration
    function_declarations = genai_tool_dict.get("function_declarations", [])
    assert len(function_declarations) > 0, "Expected at least one function declaration"
    fn_decl = function_declarations[0]

    # Check parameters
    parameters = fn_decl.get("parameters", {})  # type: ignore
    properties = parameters.get("properties", {})

    # Check location property
    assert "kind" in properties
    kind_property = properties["kind"]
    assert kind_property["type_"] == glm.Type.ARRAY

    assert "items" in kind_property
    items_property = kind_property["items"]
    assert items_property["type_"] == glm.Type.STRING
    assert items_property["enum"] == ["foo", "bar"]
