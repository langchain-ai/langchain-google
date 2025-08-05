from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    Schema,
    Tool,
    ToolConfig,
    Type,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain_google_genai._function_utils import (
    _convert_pydantic_to_genai_function,
    _format_base_tool_to_function_declaration,
    _format_dict_to_function_declaration,
    _FunctionDeclarationLike,
    _tool_choice_to_tool_config,
    convert_to_genai_function_declarations,
    replace_defs_in_schema,
    tool_to_dict,
)


def assert_property_type(
    property_dict: dict, expected_type: Type, property_name: str = "property"
) -> None:
    """
    Utility function to assert that a property has the expected Type enum value.

    Since tool_to_dict serializes Type enums to dictionaries with '_value_' field,
    this function handles the comparison correctly.

    Args:
        property_dict: The property dictionary from the serialized schema
        expected_type: The expected Type enum value
        property_name: Name of the property for error messages (optional)
    """
    actual_type_dict = property_dict.get("type", {})
    if isinstance(actual_type_dict, dict):
        actual_value = actual_type_dict.get("_value_")
        assert actual_value == expected_type.value, (
            f"Expected '{property_name}' to be {expected_type.value}, "
            f"but got {actual_value}"
        )
    else:
        # In case the type is not serialized as a dict (fallback)
        assert actual_type_dict == expected_type, (
            f"Expected '{property_name}' to be {expected_type}, "
            f"but got {actual_type_dict}"
        )


def find_any_of_option_by_type(any_of_list: list, expected_type: Type) -> dict:
    """
    Utility function to find an option in an any_of list that has the expected Type.

    Since tool_to_dict serializes Type enums to dictionaries with '_value_' field,
    this function handles the search correctly.

    Args:
        any_of_list: List of options from an any_of field
        expected_type: The Type enum value to search for

    Returns:
        The matching option dictionary

    Raises:
        AssertionError: If no option with the expected type is found
    """
    for opt in any_of_list:
        type_dict = opt.get("type", {})
        if isinstance(type_dict, dict):
            if type_dict.get("_value_") == expected_type.value:
                return opt
        else:
            if type_dict == expected_type:
                return opt

    # If we get here, no matching option was found
    available_types = []
    for opt in any_of_list:
        type_dict = opt.get("type", {})
        if isinstance(type_dict, dict):
            available_types.append(type_dict.get("_value_", "unknown"))
        else:
            available_types.append(str(type_dict))

    raise AssertionError(
        f"No option with type {expected_type.value} found in any_of. "
        f"Available types: {available_types}"
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

    assert_property_type(a_property, Type.STRING, "a")
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
    assert_property_type(items_property, Type.ARRAY, "items")
    assert items_property.get("nullable"), "Expected 'items' to be marked as nullable."
    # Check that the array items are recognized as strings

    items = items_property.get("items")
    assert isinstance(items, dict), "Expected 'items' to be a dict."

    assert_property_type(items, Type.STRING, "array items")


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

    # Check if it's OBJECT or STRING (fallback)
    actual_type_dict = data_property.get("type", {})
    if isinstance(actual_type_dict, dict):
        actual_value = actual_type_dict.get("_value_")
        assert actual_value in [
            Type.OBJECT.value,
            Type.STRING.value,
        ], f"Expected 'data' to be OBJECT or STRING, but got {actual_value}"
    else:
        assert actual_type_dict in [
            Type.OBJECT,
            Type.STRING,
        ], f"Expected 'data' to be OBJECT or STRING, but got {actual_type_dict}"
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
    assert_property_type(status_property, Type.STRING, "status")
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
search_exp = FunctionDeclaration(
    name="search",
    description="Search tool",
    parameters=Schema(
        type=Type.OBJECT,
        description="Search tool",
        properties={"question": Schema(type=Type.STRING)},
        required=["question"],
        title="search",
    ),
)


class SearchBaseTool(BaseTool):
    def _run(self) -> None:
        pass


search_base_tool = SearchBaseTool(name="search", description="Search tool")
search_base_tool_exp = FunctionDeclaration(
    name=search_base_tool.name,
    description=search_base_tool.description,
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "__arg1": Schema(type=Type.STRING),
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
search_model_exp = FunctionDeclaration(
    name="SearchModel",
    description="Search model",
    parameters=Schema(
        type=Type.OBJECT,
        description="Search model",
        properties={
            "question": Schema(type=Type.STRING),
        },
        required=["question"],
        title="SearchModel",
    ),
)

search_model_exp_pyd = FunctionDeclaration(
    name="SearchModel",
    description="Search model",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "question": Schema(type=Type.STRING),
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
    Tuple[_FunctionDeclarationLike, FunctionDeclaration, List[MagicMock], str]
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
    expected = Tool(function_declarations=fds)
    result = convert_to_genai_function_declarations(src)
    assert result == expected

    src_2 = Tool(google_search_retrieval={})
    result = convert_to_genai_function_declarations([src_2])
    assert result == src_2

    src_3: Dict[str, Any] = {"google_search_retrieval": {}}
    result = convert_to_genai_function_declarations([src_3])
    assert result == src_2

    src_4 = Tool(google_search={})
    result = convert_to_genai_function_declarations([src_4])
    assert result == src_4

    with pytest.raises(ValueError) as exc_info1:
        _ = convert_to_genai_function_declarations(["fake_tool"])  # type: ignore
    assert str(exc_info1.value).startswith("Unsupported tool")

    with pytest.raises(Exception) as exc_info:
        _ = convert_to_genai_function_declarations(
            [
                Tool(google_search_retrieval={}),
                Tool(google_search_retrieval={}),
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
    actual = tool_to_dict(convert_to_genai_function_declarations(oai_tools))[
        "function_declarations"
    ]

    # Check that we have the expected number of function declarations
    assert len(actual) == 2

    # Check the first function declaration (split_documents)
    split_docs = actual[0]
    assert split_docs["name"] == "split_documents"
    assert split_docs["description"] == "Tool."
    assert split_docs["behavior"] is None

    # Check parameters structure
    params = split_docs["parameters"]
    assert params["type"]["_value_"] == "OBJECT"
    assert params["required"] == ["chunk_size"]

    # Check properties
    properties = params["properties"]
    assert "chunk_size" in properties
    assert "chunk_overlap" in properties

    # Check chunk_size property
    chunk_size_prop = properties["chunk_size"]
    assert chunk_size_prop["type"]["_value_"] == "INTEGER"
    assert chunk_size_prop["description"] == "chunk size."
    assert chunk_size_prop["nullable"] is None

    # Check chunk_overlap property
    chunk_overlap_prop = properties["chunk_overlap"]
    assert chunk_overlap_prop["type"]["_value_"] == "INTEGER"
    assert chunk_overlap_prop["description"] == "chunk overlap."
    assert chunk_overlap_prop["nullable"] is True

    # Check the second function declaration (search_web)
    search_web_func = actual[1]
    assert search_web_func["name"] == "search_web"
    assert search_web_func["description"] == "Tool."
    assert search_web_func["behavior"] is None

    # Check parameters structure
    params = search_web_func["parameters"]
    assert params["type"]["_value_"] == "OBJECT"
    assert params["required"] == ["query"]

    # Check properties
    properties = params["properties"]
    assert "query" in properties
    assert "engine" in properties
    assert "num_results" in properties
    assert "truncate_threshold" in properties

    # Check query property
    query_prop = properties["query"]
    assert query_prop["type"]["_value_"] == "STRING"
    assert query_prop["description"] == "query."
    assert query_prop["nullable"] is None

    # Check engine property
    engine_prop = properties["engine"]
    assert engine_prop["type"]["_value_"] == "STRING"
    assert engine_prop["description"] == "engine."
    assert engine_prop["nullable"] is None

    # Check num_results property
    num_results_prop = properties["num_results"]
    assert num_results_prop["type"]["_value_"] == "INTEGER"
    assert num_results_prop["description"] == "number of results."
    assert num_results_prop["nullable"] is None

    # Check truncate_threshold property
    truncate_prop = properties["truncate_threshold"]
    assert truncate_prop["type"]["_value_"] == "INTEGER"
    assert truncate_prop["description"] == "truncate threshold."
    assert truncate_prop["nullable"] is True


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
    expected = Tool(
        function_declarations=[
            FunctionDeclaration(
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
    expected = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode=FunctionCallingConfigMode.ANY,
            allowed_function_names=["foo"],
        ),
    )
    actual = _tool_choice_to_tool_config(choice, ["foo"])
    assert expected == actual


def test_tool_to_dict_glm_tool() -> None:
    tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="multiply",
                description="Returns the product of two numbers.",
                parameters=Schema(
                    type=Type.OBJECT,
                    properties={
                        "a": Schema(type=Type.NUMBER),
                        "b": Schema(type=Type.NUMBER),
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

    # Check that we have the expected structure
    assert "function_declarations" in tool_dict
    assert len(tool_dict["function_declarations"]) == 1

    # Check the function declaration
    func_decl = tool_dict["function_declarations"][0]
    assert func_decl["name"] == "Models"
    assert func_decl["description"] is None
    assert func_decl["behavior"] is None

    # Check parameters structure
    params = func_decl["parameters"]
    assert params["type"]["_value_"] == "OBJECT"
    assert params["required"] == ["models"]

    # Check properties
    properties = params["properties"]
    assert "models" in properties

    # Check models property (array of MyModel)
    models_prop = properties["models"]
    assert models_prop["type"]["_value_"] == "ARRAY"
    assert models_prop["nullable"] is None

    # Check items of the array
    items = models_prop["items"]
    assert items["type"]["_value_"] == "OBJECT"
    assert items["description"] == "MyModel"
    assert items["required"] == ["name", "age"]

    # Check properties of MyModel
    model_properties = items["properties"]
    assert "name" in model_properties
    assert "age" in model_properties

    # Check name property
    name_prop = model_properties["name"]
    assert name_prop["type"]["_value_"] == "STRING"
    assert name_prop["nullable"] is None

    # Check age property
    age_prop = model_properties["age"]
    assert age_prop["type"]["_value_"] == "INTEGER"
    assert age_prop["nullable"] is None


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

    assert_property_type(matrix_property, Type.ARRAY, "matrix")

    items_level1 = matrix_property.get("items")
    assert isinstance(items_level1, dict), "Expected first level 'items' to be a dict."
    assert_property_type(items_level1, Type.ARRAY, "first level items")

    items_level2 = items_level1.get("items")
    assert isinstance(items_level2, dict), "Expected second level 'items' to be a dict."
    assert_property_type(items_level2, Type.STRING, "second level items")

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
    """
    Test that validates tools with Union types in function declarations
    are correctly converted to 'anyOf' in the schema.
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
    assert_property_type(helper1["properties"]["x"], Type.BOOLEAN, "x")

    # Check second option (Helper2)
    helper2 = any_of[1]
    assert isinstance(helper2, dict), "Expected second option to be a dict."
    assert "properties" in helper2, "Expected second option to have properties."
    assert "y" in helper2["properties"], "Expected second option to have 'y' property."
    assert_property_type(helper2["properties"]["y"], Type.STRING, "y")


def test_tool_with_union_primitive_types() -> None:
    """
    Test that validates tools with Union types that include primitive types
    are correctly converted to 'anyOf' in the schema.
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
    # Just verify string option exists
    _ = find_any_of_option_by_type(any_of, Type.STRING)

    # One option should be an object (Helper)
    object_option = find_any_of_option_by_type(any_of, Type.OBJECT)
    assert "properties" in object_option, "Expected object option to have properties."
    assert (
        "value" in object_option["properties"]
    ), "Expected object option to have 'value' property."
    # Note: This assertion expects the raw enum integer value (3 for NUMBER)
    # This is a special case where the test was expecting the integer value
    value_type = object_option["properties"]["value"].get("type", {})
    if isinstance(value_type, dict):
        # For serialized enum, check _value_ and convert to enum to get integer
        type_str = value_type.get("_value_")
        if type_str == "NUMBER":
            assert True, "Expected 'value' to be NUMBER."
        elif type_str == "INTEGER":
            assert True, "Expected 'value' to be INTEGER."
        else:
            assert False, f"Expected 'value' to be NUMBER or INTEGER, got {type_str}"
    else:
        assert (
            value_type == 3
        ), f"Expected 'value' to be NUMBER or INTEGER (3), got {value_type}"


def test_tool_with_nested_union_types() -> None:
    """
    Test that validates tools with nested Union types are correctly converted
    to nested 'anyOf' structures in the schema.
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
        contacts: List[Union[str, Contact]] = []

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
    # Just verify string option exists
    _ = find_any_of_option_by_type(location_any_of, Type.STRING)

    # One option should be an object (Address)
    address_option = find_any_of_option_by_type(location_any_of, Type.OBJECT)
    assert "properties" in address_option, "Expected address option to have properties"
    assert (
        "city" in address_option["properties"]
    ), "Expected Address to have 'city' property."


def test_tool_invocation_with_union_types() -> None:
    """
    Test that validates tools with Union types can be correctly invoked
    with either type from the union.
    """

    class Configuration(BaseModel):
        """Configuration model."""

        settings: Dict[str, str] = {}

    @tool
    def configure_service(service_name: str, config: Union[str, Configuration]) -> str:
        """
        Configure a service with either a configuration string or object.

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
    assert Type.STRING in type_variants, "Expected STRING to be one of the variants"
    assert Type.OBJECT in type_variants, "Expected OBJECT to be one of the variants"

    # Find the object variant
    object_variant = None
    for option in config_property.any_of:
        if option.type == Type.OBJECT:
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
    """
    Test that validates Field with Union types in Pydantic models
    are correctly converted to 'anyOf' in the schema.
    """

    class Helper1(BaseModel):
        """Helper class 1."""

        x: bool = False

    class Helper2(BaseModel):
        """Helper class 2."""

        y: str = "1"

    class GetWeather(BaseModel):
        """
        Get weather information for a location.
        """

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
    assert (
        "description" in location_property
    ), "Expected description field in location property"
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
