from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import google.ai.generativelanguage as glm
import pytest
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from typing_extensions import Annotated

from langchain_google_genai._function_utils import (
    _convert_pydantic_to_genai_function,
    _format_base_tool_to_function_declaration,
    _format_dict_to_function_declaration,
    _FunctionDeclarationLike,
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


# reusable test inputs
def search(question: str) -> str:
    """Search tool"""
    return question


search_tool = tool(search)
search_exp = glm.FunctionDeclaration(
    name="search",
    description="Search tool",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        description="Search tool",
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
    """Search model"""

    question: str


search_model_schema = SearchModel.model_json_schema()
search_model_dict = {
    "name": search_model_schema["title"],
    "description": search_model_schema["description"],
    "parameters": search_model_schema,
}
search_model_exp = glm.FunctionDeclaration(
    name="SearchModel",
    description="Search model",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        description="Search model",
        properties={
            "question": glm.Schema(type=glm.Type.STRING),
        },
        required=["question"],
        title="SearchModel",
    ),
)

search_model_exp_pyd = glm.FunctionDeclaration(
    name="SearchModel",
    description="Search model",
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

SRC_EXP_MOCKS_DESC: List[
    Tuple[_FunctionDeclarationLike, glm.FunctionDeclaration, List[MagicMock], str]
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

    src = [src for src, _, _, _ in SRC_EXP_MOCKS_DESC]
    fds = [fd for _, fd, _, _ in SRC_EXP_MOCKS_DESC]
    expected = glm.Tool(function_declarations=fds)
    result = convert_to_genai_function_declarations(src)
    assert result == expected

    src_2 = glm.Tool(google_search_retrieval={})
    result = convert_to_genai_function_declarations([src_2])
    assert result == src_2

    src_3: Dict[str, Any] = {"google_search_retrieval": {}}
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
                "any_of": [],
                "type_": 6,
                "properties": {
                    "chunk_overlap": {
                        "any_of": [],
                        "type_": 3,
                        "description": "chunk overlap.",
                        "format_": "",
                        "nullable": True,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "property_ordering": [],
                        "required": [],
                        "title": "",
                    },
                    "chunk_size": {
                        "any_of": [],
                        "type_": 3,
                        "description": "chunk size.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "property_ordering": [],
                        "required": [],
                        "title": "",
                    },
                },
                "property_ordering": [],
                "required": ["chunk_size"],
                "title": "",
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
                "any_of": [],
                "type_": 6,
                "properties": {
                    "truncate_threshold": {
                        "any_of": [],
                        "type_": 3,
                        "description": "truncate threshold.",
                        "format_": "",
                        "nullable": True,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "property_ordering": [],
                        "properties": {},
                        "required": [],
                        "title": "",
                    },
                    "query": {
                        "any_of": [],
                        "type_": 1,
                        "description": "query.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "property_ordering": [],
                        "required": [],
                        "title": "",
                    },
                    "engine": {
                        "any_of": [],
                        "type_": 1,
                        "description": "engine.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "property_ordering": [],
                        "properties": {},
                        "required": [],
                        "title": "",
                    },
                    "num_results": {
                        "any_of": [],
                        "type_": 3,
                        "description": "number of results.",
                        "format_": "",
                        "nullable": False,
                        "enum": [],
                        "max_items": "0",
                        "min_items": "0",
                        "properties": {},
                        "property_ordering": [],
                        "required": [],
                        "title": "",
                    },
                },
                "property_ordering": [],
                "required": ["query"],
                "title": "",
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
                    "any_of": [],
                    "description": "",
                    "enum": [],
                    "format_": "",
                    "max_items": "0",
                    "min_items": "0",
                    "nullable": False,
                    "properties": {
                        "models": {
                            "any_of": [],
                            "description": "",
                            "enum": [],
                            "format_": "",
                            "items": {
                                "any_of": [],
                                "description": "MyModel",
                                "enum": [],
                                "format_": "",
                                "max_items": "0",
                                "min_items": "0",
                                "nullable": False,
                                "properties": {
                                    "age": {
                                        "any_of": [],
                                        "description": "",
                                        "enum": [],
                                        "format_": "",
                                        "max_items": "0",
                                        "min_items": "0",
                                        "nullable": False,
                                        "properties": {},
                                        "property_ordering": [],
                                        "required": [],
                                        "title": "",
                                        "type_": 3,
                                    },
                                    "name": {
                                        "any_of": [],
                                        "description": "",
                                        "enum": [],
                                        "format_": "",
                                        "max_items": "0",
                                        "min_items": "0",
                                        "nullable": False,
                                        "properties": {},
                                        "property_ordering": [],
                                        "required": [],
                                        "title": "",
                                        "type_": 1,
                                    },
                                },
                                "property_ordering": [],
                                "required": ["name", "age"],
                                "title": "",
                                "type_": 6,
                            },
                            "max_items": "0",
                            "min_items": "0",
                            "nullable": False,
                            "properties": {},
                            "property_ordering": [],
                            "required": [],
                            "title": "",
                            "type_": 5,
                        }
                    },
                    "property_ordering": [],
                    "required": ["models"],
                    "title": "",
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


def test_tool_with_doubly_nested_list_param() -> None:
    """
    Tests a tool parameter with a doubly nested list (List[List[str]]),
    verifying that the GAPIC schema correctly represents the nested items.
    """

    @tool(parse_docstring=True)
    def process_nested_data(
        matrix: List[List[str]],
    ) -> str:
        """
        Processes a matrix (list of lists of strings).

        Args:
          matrix: The nested list data.
        """
        return f"Processed {len(matrix)} rows."

    oai_tool = convert_to_openai_tool(process_nested_data)

    genai_tool = convert_to_genai_function_declarations([oai_tool])

    genai_tool_dict = tool_to_dict(genai_tool)
    assert isinstance(genai_tool_dict, dict), "Expected a dict."

    function_declarations = genai_tool_dict.get("function_declarations")
    assert isinstance(function_declarations, list) and len(function_declarations) == 1
    fn_decl = function_declarations[0]
    assert isinstance(fn_decl, dict)

    parameters = fn_decl.get("parameters")
    assert isinstance(parameters, dict)

    properties = parameters.get("properties")
    assert isinstance(properties, dict)

    matrix_property = properties.get("matrix")
    assert isinstance(matrix_property, dict)

    assert (
        matrix_property.get("type_") == glm.Type.ARRAY
    ), "Expected 'matrix' to be ARRAY."

    items_level1 = matrix_property.get("items")
    assert isinstance(items_level1, dict), "Expected first level 'items' to be a dict."
    assert (
        items_level1.get("type_") == glm.Type.ARRAY
    ), "Expected first level items to be ARRAY."

    items_level2 = items_level1.get("items")
    assert isinstance(items_level2, dict), "Expected second level 'items' to be a dict."
    assert (
        items_level2.get("type_") == glm.Type.STRING
    ), "Expected second level items to be STRING."

    assert "description" in matrix_property
    assert "description" in items_level1
    assert "description" in items_level2
