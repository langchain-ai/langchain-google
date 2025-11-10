from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from enum import Enum
from typing import Any, cast
from unittest.mock import Mock, patch

import google.cloud.aiplatform_v1beta1.types as gapic
import pytest
import vertexai.generative_models as vertexai  # TODO: migrate to google-genai
from google.cloud.aiplatform_v1beta1.types import (
    FunctionCallingConfig as GapicFunctionCallingConfig,
)
from google.cloud.aiplatform_v1beta1.types import (
    ToolConfig as GapicToolConfig,
)
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    convert_to_openai_tool,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel, Field
from pydantic.v1 import (
    BaseModel as BaseModelV1,
)
from pydantic.v1 import (
    Field as FieldV1,
)

from langchain_google_vertexai.functions_utils import (
    _format_base_tool_to_function_declaration,
    _format_dict_to_function_declaration,
    _format_json_schema_to_gapic,
    _format_json_schema_to_gapic_v1,
    _format_pydantic_to_function_declaration,
    _format_to_gapic_function_declaration,
    _format_to_gapic_tool,
    _format_tool_config,
    _format_vertex_to_function_declaration,
    _FunctionDeclarationLike,
    _tool_choice_to_tool_config,
    _ToolType,
)


def test_format_json_schema_to_gapic() -> None:
    # Simple case
    class RecordPerson(BaseModel):
        """Record some identifying information about a person."""

        name: str
        age: int | None

    schema = RecordPerson.model_json_schema()
    result = _format_json_schema_to_gapic(schema)
    expected = {
        "title": "RecordPerson",
        "type": "OBJECT",
        "description": "Record some identifying information about a person.",
        "properties": {
            "name": {"title": "Name", "type": "STRING"},
            "age": {"type": "INTEGER", "title": "Age"},
        },
        "required": ["name"],
    }
    assert result == expected

    # Nested case
    class StringEnum(str, Enum):
        pear = "pear"
        banana = "banana"

    class A(BaseModel):
        """Class A."""

        int_field: int | None

    class B(BaseModel):
        object_field: A | None = Field(description="Class A")
        array_field: Sequence[A]
        int_field: int = Field(description="int field", ge=1, le=10)
        str_field: str = Field(
            min_length=1,
            max_length=10,
            pattern="^[A-Z]{1,10}$",
            json_schema_extra={"example": "ABCD"},
        )
        str_enum_field: StringEnum

    schema = B.model_json_schema()
    result = _format_json_schema_to_gapic(dereference_refs(schema))

    expected = {
        "properties": {
            "object_field": {
                "description": "Class A",
                "properties": {"int_field": {"type": "INTEGER", "title": "Int Field"}},
                "required": [],
                "title": "A",
                "type": "OBJECT",
            },
            "array_field": {
                "items": {
                    "description": "Class A.",
                    "properties": {
                        "int_field": {"type": "INTEGER", "title": "Int Field"}
                    },
                    "required": [],
                    "title": "A",
                    "type": "OBJECT",
                },
                "type": "ARRAY",
                "title": "Array Field",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10,
                "minimum": 1,
                "title": "Int Field",
                "type": "INTEGER",
            },
            "str_field": {
                "example": "ABCD",
                "maxLength": 10,
                "minLength": 1,
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
                "type": "STRING",
            },
            "str_enum_field": {
                "enum": ["pear", "banana"],
                "title": "StringEnum",
                "type": "STRING",
            },
        },
        "type": "OBJECT",
        "title": "B",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
    assert result == expected

    gapic_schema = cast("gapic.Schema", gapic.Schema.from_json(json.dumps(result)))
    assert gapic_schema.type_ == gapic.Type.OBJECT
    assert gapic_schema.title == expected["title"]
    assert gapic_schema.required == expected["required"]
    assert (
        gapic_schema.properties["str_field"].example
        == expected["properties"]["str_field"]["example"]  # type: ignore
    )


# Move class definitions outside function to avoid forward reference issues in Pydantic
# V1
class _RecordPersonV1(BaseModelV1):
    """Record some identifying information about a person."""

    name: str
    age: int | None


class _StringEnumV1(str, Enum):
    pear = "pear"
    banana = "banana"


class _AV1(BaseModelV1):
    """Class A."""

    int_field: int | None


class _BV1(BaseModelV1):
    object_field: _AV1 | None = FieldV1(description="Class A")
    array_field: Sequence[_AV1]
    int_field: int = FieldV1(description="int field", minimum=1, maximum=10)
    str_field: str = FieldV1(
        min_length=1, max_length=10, pattern="^[A-Z]{1,10}$", example="ABCD"
    )
    str_enum_field: _StringEnumV1


def test_format_json_schema_to_gapic_v1() -> None:
    # Simple case
    schema = _RecordPersonV1.schema()
    result = _format_json_schema_to_gapic_v1(schema)
    expected = {
        "title": "_RecordPersonV1",
        "type": "OBJECT",
        "description": "Record some identifying information about a person.",
        "properties": {
            "name": {"title": "Name", "type": "STRING"},
            "age": {"type": "INTEGER", "title": "Age"},
        },
        "required": ["name"],
    }
    assert result == expected

    # Nested case
    schema = _BV1.schema()
    result = _format_json_schema_to_gapic_v1(dereference_refs(schema))

    expected = {
        "properties": {
            "object_field": {
                "description": "Class A.",
                "properties": {"int_field": {"type": "INTEGER", "title": "Int Field"}},
                "title": "_AV1",
                "type": "OBJECT",
            },
            "array_field": {
                "items": {
                    "description": "Class A.",
                    "properties": {
                        "int_field": {"type": "INTEGER", "title": "Int Field"}
                    },
                    "title": "_AV1",
                    "type": "OBJECT",
                },
                "type": "ARRAY",
                "title": "Array Field",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10,
                "minimum": 1,
                "title": "Int Field",
                "type": "INTEGER",
            },
            "str_field": {
                "example": "ABCD",
                "maxLength": 10,
                "minLength": 1,
                "pattern": "^[A-Z]{1,10}$",
                "title": "Str Field",
                "type": "STRING",
            },
            "str_enum_field": {
                "description": "An enumeration.",
                "enum": ["pear", "banana"],
                "title": "_StringEnumV1",
                "type": "STRING",
            },
        },
        "type": "OBJECT",
        "title": "_BV1",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
    assert result == expected

    gapic_schema = cast("gapic.Schema", gapic.Schema.from_json(json.dumps(result)))
    assert gapic_schema.type_ == gapic.Type.OBJECT
    assert gapic_schema.title == expected["title"]
    assert gapic_schema.required == expected["required"]
    assert (
        gapic_schema.properties["str_field"].example
        == expected["properties"]["str_field"]["example"]  # type: ignore
    )


def test_format_json_schema_to_gapic_union_types() -> None:
    """Test that union types are consistent between v1 and v2."""

    class RecordPerson_v1(BaseModelV1):
        name: str
        age: int | str

    class RecordPerson(BaseModel):
        name: str
        age: int | str

    schema_v1 = RecordPerson_v1.schema()
    schema_v2 = RecordPerson.model_json_schema()

    result_v1 = _format_json_schema_to_gapic_v1(schema_v1)
    result_v1["title"] = "RecordPerson"

    result_v2 = _format_json_schema_to_gapic(schema_v2)

    assert result_v1 == result_v2


# reusable test inputs
def search(question: str) -> str:
    """Search tool."""
    return question


search_tool = tool(search)
search_exp = gapic.FunctionDeclaration(
    name="search",
    description="Search tool.",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        description="Search tool.",
        title="search",
        properties={"question": gapic.Schema(type=gapic.Type.STRING, title="Question")},
        required=["question"],
    ),
)

search_vfd = vertexai.FunctionDeclaration.from_func(search)
search_vfd_exp = gapic.FunctionDeclaration(
    name="search",
    description="Search tool.",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        title="search",
        description="Search tool.",
        properties={"question": gapic.Schema(type=gapic.Type.STRING, title="Question")},
        required=["question"],
        property_ordering=["question"],
    ),
)


class SearchBaseTool(BaseTool):
    def _run(self) -> None:
        pass


search_base_tool = SearchBaseTool(name="search", description="Search tool.")
search_base_tool_exp = gapic.FunctionDeclaration(
    name=search_base_tool.name,
    description=search_base_tool.description,
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        properties={
            "__arg1": gapic.Schema(type=gapic.Type.STRING),
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
search_model_exp = gapic.FunctionDeclaration(
    name="SearchModel",
    description="Search model.",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        title="SearchModel",
        description="Search model.",
        properties={
            "question": gapic.Schema(type=gapic.Type.STRING, title="Question"),
        },
        required=["question"],
    ),
)

mock_dict = Mock(name="mock_dicts", wraps=_format_dict_to_function_declaration)
mock_base_tool = Mock(
    name="mock_base_tool", wraps=_format_base_tool_to_function_declaration
)
mock_pydantic = Mock(
    name="mock_pydantic", wraps=_format_pydantic_to_function_declaration
)
mock_vertex = Mock("mock_vertex", wraps=_format_vertex_to_function_declaration)

TO_FUNCTION_DECLARATION_MOCKS = [mock_dict, mock_base_tool, mock_pydantic, mock_vertex]

SRC_EXP_MOCKS_DESC: list[
    tuple[_FunctionDeclarationLike, gapic.FunctionDeclaration, list[Mock], str]
] = [
    (search, search_exp, [mock_base_tool], "plain function"),
    (search_tool, search_exp, [mock_base_tool], "LC tool"),
    (search_base_tool, search_base_tool_exp, [mock_base_tool], "LC base tool"),
    (search_vfd, search_vfd_exp, [mock_vertex, mock_dict], "Vertex FD"),
    (SearchModel, search_model_exp, [mock_pydantic], "Pydantic model"),
    (search_model_dict, search_model_exp, [mock_dict], "dict"),
]


@patch(
    "langchain_google_vertexai.functions_utils._format_vertex_to_function_declaration",
    new=mock_vertex,
)
@patch(
    "langchain_google_vertexai.functions_utils._format_pydantic_to_function_declaration",
    new=mock_pydantic,
)
@patch(
    "langchain_google_vertexai.functions_utils._format_base_tool_to_function_declaration",
    new=mock_base_tool,
)
@patch(
    "langchain_google_vertexai.functions_utils._format_dict_to_function_declaration",
    new=mock_dict,
)
def test_format_to_gapic_function_declaration() -> None:
    for src, exp, mocks, desc in SRC_EXP_MOCKS_DESC:
        res = _format_to_gapic_function_declaration(src)
        assert res == exp
        for m in TO_FUNCTION_DECLARATION_MOCKS:
            if m in mocks:
                assert m.called, (
                    f"Mock {m._extract_mock_name()} should be called"
                    f" for {desc}, but it wasn't"
                )
            else:
                assert not m.called, (
                    f"Mock {m._extract_mock_name()} should not be called"
                    f"for {desc}, but it was"
                )
            m.reset_mock()


def test_format_to_gapic_tool() -> None:
    src: list[_FunctionDeclarationLike] = [src for src, _, _, _ in SRC_EXP_MOCKS_DESC]
    fds = [fd for _, fd, _, _ in SRC_EXP_MOCKS_DESC]
    expected = gapic.Tool(function_declarations=fds)
    result = _format_to_gapic_tool(src)
    assert result == expected

    additional_tools: list[_ToolType] = [
        gapic.Tool(function_declarations=[search_model_exp]),
        vertexai.Tool.from_function_declarations(
            [vertexai.FunctionDeclaration.from_func(search)]
        ),
        {"function_declarations": [search_model_dict]},
    ]
    src_2 = src + additional_tools
    expected = gapic.Tool(
        function_declarations=[*fds, search_model_exp, search_vfd_exp, search_model_exp]
    )
    result = _format_to_gapic_tool(src_2)
    assert result == expected

    src_3 = gapic.Tool(google_search_retrieval={})
    result = _format_to_gapic_tool([src_3])
    assert result == src_3

    src_4: dict[str, Any] = {"google_search_retrieval": {}}
    result = _format_to_gapic_tool([src_4])
    assert result == src_3

    src_5 = gapic.Tool(
        retrieval=gapic.Retrieval(
            vertex_ai_search=gapic.VertexAISearch(datastore="datastore")
        )
    )

    result = _format_to_gapic_tool([src_5])
    assert result == src_5

    src_6 = {
        "retrieval": {
            "vertex_ai_search": {
                "datastore": "datastore",
            }
        }
    }
    result = _format_to_gapic_tool([src_6])
    assert result == src_5

    with pytest.raises(ValueError) as exc_info1:
        # type ignore since we're testing invalid input
        _ = _format_to_gapic_tool(["fake_tool"])  # type: ignore[list-item]
    assert str(exc_info1.value).startswith("Unsupported tool")

    with pytest.raises(Exception) as exc_info:
        _ = _format_to_gapic_tool(
            [
                gapic.Tool(function_declarations=[search_model_exp]),
                gapic.Tool(google_search_retrieval={}),
                gapic.Tool(
                    retrieval=gapic.Retrieval(
                        vertex_ai_search=gapic.VertexAISearch(datastore="datastore")
                    )
                ),
            ]
        )
    assert str(exc_info.value).startswith(
        "Providing multiple retrieval, google_search_retrieval"
    )

    with pytest.raises(Exception) as exc_info:
        _ = _format_to_gapic_tool(
            [
                gapic.Tool(google_search_retrieval={}),
                gapic.Tool(google_search_retrieval={}),
            ]
        )
    assert str(exc_info.value).startswith(
        "Providing multiple retrieval, google_search_retrieval"
    )


def test_format_tool_config_invalid() -> None:
    with pytest.raises(ValueError):
        _format_tool_config({})  # type: ignore


def test_format_tool_config() -> None:
    tool_config = _format_tool_config(
        {
            "function_calling_config": {
                "mode": gapic.FunctionCallingConfig.Mode.ANY,
                "allowed_function_names": ["my_fun"],
            }
        }
    )
    assert tool_config == gapic.ToolConfig(
        function_calling_config=gapic.FunctionCallingConfig(
            mode=gapic.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=["my_fun"],
        )
    )


@pytest.mark.parametrize(
    "choice",
    [True, "foo", ["foo"], "any", {"type": "function", "function": {"name": "foo"}}],
)
def test__tool_choice_to_tool_config(choice: Any) -> None:
    expected = GapicToolConfig(
        function_calling_config=GapicFunctionCallingConfig(
            mode=gapic.FunctionCallingConfig.Mode.ANY,
            allowed_function_names=["foo"],
        ),
    )
    actual = _tool_choice_to_tool_config(choice, ["foo"])
    assert expected == actual


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10 or higher")
def test_nested_bind_tools() -> None:
    class Person(BaseModel):
        name: str = Field(description="The name.")
        hair_color: str | None = Field("Hair color, only if provided.")  # type: ignore[syntax, unused-ignore]

    class People(BaseModel):
        group_ids: list[int] = Field(description="The group ids.")
        data: list[Person] = Field(description="The people.")

    tool = convert_to_openai_tool(People)
    function = convert_to_openai_tool(cast("dict", tool))["function"]
    converted_tool = _format_dict_to_function_declaration(
        cast("FunctionDescription", function)
    )
    assert converted_tool.name == "People"


def test_tool_with_union_types() -> None:
    """Test that validates tools with Union types in function declarations
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
        date: Helper1 | Helper2 = Helper1()

    # Convert the model schema
    schema = GetWeather.model_json_schema()
    dereferenced_schema = dereference_refs(schema)
    result = _format_json_schema_to_gapic(dereferenced_schema)

    # Check that the properties exist
    assert "properties" in result
    assert "location" in result["properties"]
    assert "date" in result["properties"]

    # Verify that anyOf is present in the date property
    date_property = result["properties"]["date"]
    assert "anyOf" in date_property
    assert isinstance(date_property["anyOf"], list)
    assert len(date_property["anyOf"]) == 2

    # Check first option (Helper1)
    helper1 = date_property["anyOf"][0]
    assert "properties" in helper1
    assert "x" in helper1["properties"]
    assert helper1["properties"]["x"]["type"] == "BOOLEAN"

    # Check second option (Helper2)
    helper2 = date_property["anyOf"][1]
    assert "properties" in helper2
    assert "y" in helper2["properties"]
    assert helper2["properties"]["y"]["type"] == "STRING"

    # Test with conversion to gapic.Schema
    gapic_schema = gapic.Schema.from_json(json.dumps(result))
    date_prop = gapic_schema.properties["date"]
    assert hasattr(date_prop, "any_of")
    assert len(date_prop.any_of) == 2


def test_tool_with_union_primitive_types() -> None:
    """Test that validates tools with Union types that include primitive types
    are correctly converted to 'anyOf' in the schema.
    """

    class Helper(BaseModel):
        """Test helper class."""

        value: int = 42

    class SearchQuery(BaseModel):
        """Search query model with a union parameter."""

        query: str = "default query"
        filter: str | Helper = "default filter"

    # Convert the model schema
    schema = SearchQuery.model_json_schema()
    dereferenced_schema = dereference_refs(schema)
    result = _format_json_schema_to_gapic(dereferenced_schema)

    # Check that the properties exist
    assert "properties" in result
    assert "filter" in result["properties"]

    # Verify that anyOf is present in the filter property
    filter_property = result["properties"]["filter"]
    assert "anyOf" in filter_property
    assert isinstance(filter_property["anyOf"], list)
    assert len(filter_property["anyOf"]) == 2

    # One option should be a string
    string_option = next(
        (opt for opt in filter_property["anyOf"] if opt.get("type") == "STRING"), None
    )
    assert string_option is not None

    # One option should be an object (Helper)
    object_option = next(
        (opt for opt in filter_property["anyOf"] if opt.get("type") == "OBJECT"), None
    )
    assert object_option is not None
    assert "properties" in object_option
    assert "value" in object_option["properties"]
    assert object_option["properties"]["value"]["type"] == "INTEGER"

    # Test with conversion to gapic.Schema
    gapic_schema = gapic.Schema.from_json(json.dumps(result))
    filter_prop = gapic_schema.properties["filter"]
    assert hasattr(filter_prop, "any_of")
    assert len(filter_prop.any_of) == 2


def test_tool_with_nested_union_types() -> None:
    """Test that validates tools with nested Union types are correctly converted
    to nested 'anyOf' structures in the schema.
    """

    class Address(BaseModel):
        """Address model."""

        street: str = "123 Main St"
        city: str = "Anytown"

    class Contact(BaseModel):
        """Contact model."""

        email: str = "user@example.com"
        phone: str | None = None

    class Person(BaseModel):
        """Person model with complex nested unions."""

        name: str
        location: str | Address = "Unknown"
        contacts: list[str | Contact] = []

    # Convert the model schema
    schema = Person.model_json_schema()
    dereferenced_schema = dereference_refs(schema)
    result = _format_json_schema_to_gapic(dereferenced_schema)

    # Check that the properties exist
    assert "properties" in result
    assert "name" in result["properties"]
    assert "location" in result["properties"]
    assert "contacts" in result["properties"]

    # Check location property (direct Union)
    location_property = result["properties"]["location"]
    assert "anyOf" in location_property
    location_any_of = location_property["anyOf"]
    assert len(location_any_of) == 2

    # One option should be a string
    string_option = next(
        (opt for opt in location_any_of if opt.get("type") == "STRING"), None
    )
    assert string_option is not None

    # One option should be an object (Address)
    address_option = next(
        (opt for opt in location_any_of if opt.get("type") == "OBJECT"), None
    )
    assert address_option is not None
    assert "properties" in address_option
    assert "city" in address_option["properties"]

    # Check contacts property (List of Union types)
    contacts_property = result["properties"]["contacts"]
    assert "type" in contacts_property
    assert contacts_property["type"] == "ARRAY"
    assert "items" in contacts_property

    # The items should have anyOf for the union types
    items = contacts_property["items"]
    assert "anyOf" in items
    assert len(items["anyOf"]) == 2

    # Convert to gapic.Schema to ensure it's valid
    gapic_schema = gapic.Schema.from_json(json.dumps(result))
    assert gapic_schema.properties["location"].any_of is not None
    assert len(gapic_schema.properties["location"].any_of) == 2


def test_tool_field_union_types() -> None:
    """Test that validates Field with Union types in Pydantic models
    are correctly converted to 'anyOf' in the schema.
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
        date: Helper1 | Helper2 = Field(description="Test field")

    # Convert the model schema
    schema = GetWeather.model_json_schema()
    dereferenced_schema = dereference_refs(schema)
    result = _format_json_schema_to_gapic(dereferenced_schema)

    # Check that the properties exist
    assert "properties" in result
    assert "location" in result["properties"]
    assert "date" in result["properties"]

    # Check location property
    location_property = result["properties"]["location"]
    assert "description" in location_property
    assert (
        location_property["description"] == "The city and country, e.g. New York, USA"
    )

    # Check date property (the union type)
    date_property = result["properties"]["date"]
    assert "anyOf" in date_property
    assert "description" in date_property
    assert date_property["description"] == "Test field"

    any_of = date_property["anyOf"]
    assert len(any_of) == 2

    # Extract the titles of the models in the anyOf
    model_titles = []
    for option in any_of:
        if "title" in option:
            model_titles.append(option["title"])

    assert "Helper1" in model_titles
    assert "Helper2" in model_titles

    # Check that the required fields include both location and date
    assert "required" in result
    required_fields = result["required"]
    assert "location" in required_fields
    assert "date" in required_fields

    # Convert to gapic.Schema to ensure it's valid
    gapic_schema = gapic.Schema.from_json(json.dumps(result))
    date_prop = gapic_schema.properties["date"]
    assert hasattr(date_prop, "any_of")
    assert len(date_prop.any_of) == 2


def test_union_nullable_types() -> None:
    """Test that validates the handling of Union types with null (None/Optional)
    are correctly handled by removing them from required fields.
    """

    class Config(BaseModel):
        """Config model with nullable fields."""

        required_field: str
        optional_primitive: int | None = None
        optional_complex: dict[str, str] | None = None

    schema = Config.model_json_schema()
    dereferenced_schema = dereference_refs(schema)
    result = _format_json_schema_to_gapic(dereferenced_schema)

    # Check that only the required_field is in required
    assert "required" in result
    assert "required_field" in result["required"]
    assert "optional_primitive" not in result["required"]
    assert "optional_complex" not in result["required"]

    # Check that the nullable fields have the correct schema
    assert "properties" in result

    # Optional primitive field should have INTEGER type (not anyOf)
    assert "optional_primitive" in result["properties"]
    optional_primitive = result["properties"]["optional_primitive"]
    assert "type" in optional_primitive
    assert optional_primitive["type"] == "INTEGER"

    # Optional complex field should have OBJECT type
    assert "optional_complex" in result["properties"]
    optional_complex = result["properties"]["optional_complex"]
    assert "type" in optional_complex
    assert optional_complex["type"] == "OBJECT"

    # Convert to gapic.Schema to ensure it's valid
    gapic_schema = gapic.Schema.from_json(json.dumps(result))
    assert "required_field" in gapic_schema.required
    assert "optional_primitive" not in gapic_schema.required
    assert "optional_complex" not in gapic_schema.required
