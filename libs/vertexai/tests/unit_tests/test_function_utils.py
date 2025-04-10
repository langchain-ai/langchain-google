import json
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from unittest.mock import Mock, patch

import google.cloud.aiplatform_v1beta1.types as gapic
import pytest
import vertexai.generative_models as vertexai  # type: ignore
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
)


def test_format_json_schema_to_gapic():
    # Simple case
    class RecordPerson(BaseModel):
        """Record some identifying information about a person."""

        name: str
        age: Optional[int]

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
        """Class A"""

        int_field: Optional[int]

    class B(BaseModel):
        object_field: Optional[A] = Field(description="Class A")
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
                    "description": "Class A",
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

    gapic_schema = cast(gapic.Schema, gapic.Schema.from_json(json.dumps(result)))
    assert gapic_schema.type_ == gapic.Type.OBJECT
    assert gapic_schema.title == expected["title"]
    assert gapic_schema.required == expected["required"]
    assert (
        gapic_schema.properties["str_field"].example
        == expected["properties"]["str_field"]["example"]  # type: ignore
    )


def test_format_json_schema_to_gapic_v1():
    # Simple case
    class RecordPerson(BaseModelV1):
        """Record some identifying information about a person."""

        name: str
        age: Optional[int]

    schema = RecordPerson.schema()
    result = _format_json_schema_to_gapic_v1(schema)
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

    class A(BaseModelV1):
        """Class A"""

        int_field: Optional[int]

    class B(BaseModelV1):
        object_field: Optional[A] = FieldV1(description="Class A")
        array_field: Sequence[A]
        int_field: int = FieldV1(description="int field", minimum=1, maximum=10)
        str_field: str = FieldV1(
            min_length=1, max_length=10, pattern="^[A-Z]{1,10}$", example="ABCD"
        )
        str_enum_field: StringEnum

    schema = B.schema()
    result = _format_json_schema_to_gapic_v1(dereference_refs(schema))

    expected = {
        "properties": {
            "object_field": {
                "description": "Class A",
                "properties": {"int_field": {"type": "INTEGER", "title": "Int Field"}},
                "title": "A",
                "type": "OBJECT",
            },
            "array_field": {
                "items": {
                    "description": "Class A",
                    "properties": {
                        "int_field": {"type": "INTEGER", "title": "Int Field"}
                    },
                    "title": "A",
                    "type": "OBJECT",
                },
                "type": "ARRAY",
                "title": "Array Field",
            },
            "int_field": {
                "description": "int field",
                "maximum": 10.0,
                "minimum": 1.0,
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
                "title": "StringEnum",
                "type": "STRING",
            },
        },
        "type": "OBJECT",
        "title": "B",
        "required": ["array_field", "int_field", "str_field", "str_enum_field"],
    }
    assert result == expected

    gapic_schema = cast(gapic.Schema, gapic.Schema.from_json(json.dumps(result)))
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
        age: Union[int, str]

    class RecordPerson(BaseModel):
        name: str
        age: Union[int, str]

    schema_v1 = RecordPerson_v1.schema()
    schema_v2 = RecordPerson.model_json_schema()
    del schema_v2

    result_v1 = _format_json_schema_to_gapic_v1(schema_v1)
    # result_v2 = _format_json_schema_to_gapic(schema_v2)
    result_v1["title"] = "RecordPerson"

    # TODO: add a proper support for Union since it has finally arrived!
    # assert result_v1 == result_v2


# reusable test inputs
def search(question: str) -> str:
    """Search tool"""
    return question


search_tool = tool(search)
search_exp = gapic.FunctionDeclaration(
    name="search",
    description="Search tool",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        description="Search tool",
        title="search",
        properties={"question": gapic.Schema(type=gapic.Type.STRING, title="Question")},
        required=["question"],
    ),
)

search_vfd = vertexai.FunctionDeclaration.from_func(search)
search_vfd_exp = gapic.FunctionDeclaration(
    name="search",
    description="Search tool",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        title="search",
        description="Search tool",
        properties={"question": gapic.Schema(type=gapic.Type.STRING, title="Question")},
        required=["question"],
        property_ordering=["question"],
    ),
)


class SearchBaseTool(BaseTool):
    def _run(self):
        pass


search_base_tool = SearchBaseTool(name="search", description="Search tool")
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
    """Search model"""

    question: str


search_model_schema = SearchModel.model_json_schema()
search_model_dict = {
    "name": search_model_schema["title"],
    "description": search_model_schema["description"],
    "parameters": search_model_schema,
}
search_model_exp = gapic.FunctionDeclaration(
    name="SearchModel",
    description="Search model",
    parameters=gapic.Schema(
        type=gapic.Type.OBJECT,
        title="SearchModel",
        description="Search model",
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

SRC_EXP_MOCKS_DESC: List[
    Tuple[_FunctionDeclarationLike, gapic.FunctionDeclaration, List[Mock], str]
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
def test_format_to_gapic_function_declaration():
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


def test_format_to_gapic_tool():
    src = [src for src, _, _, _ in SRC_EXP_MOCKS_DESC]
    fds = [fd for _, fd, _, _ in SRC_EXP_MOCKS_DESC]
    expected = gapic.Tool(function_declarations=fds)
    result = _format_to_gapic_tool(src)
    assert result == expected

    src_2 = src + [
        gapic.Tool(function_declarations=[search_model_exp]),
        vertexai.Tool.from_function_declarations(
            [vertexai.FunctionDeclaration.from_func(search)]
        ),
        {"function_declarations": [search_model_dict]},
    ]
    expected = gapic.Tool(
        function_declarations=fds + [search_model_exp, search_vfd_exp, search_model_exp]
    )
    result = _format_to_gapic_tool(src_2)
    assert result == expected

    src_3 = gapic.Tool(google_search_retrieval={})
    result = _format_to_gapic_tool([src_3])
    assert result == src_3

    src_4: Dict[str, Any] = {"google_search_retrieval": {}}
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
        _ = _format_to_gapic_tool(["fake_tool"])
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


def test_format_tool_config_invalid():
    with pytest.raises(ValueError):
        _format_tool_config({})  # type: ignore


def test_format_tool_config():
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
    (True, "foo", ["foo"], "any", {"type": "function", "function": {"name": "foo"}}),
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
def test_nested_bind_tools():
    class Person(BaseModel):
        name: str = Field(description="The name.")
        hair_color: str | None = Field("Hair color, only if provided.")  # type: ignore[syntax, unused-ignore]

    class People(BaseModel):
        data: list[Person] = Field(description="The people.")

    tool = convert_to_openai_tool(People)
    function = convert_to_openai_tool(cast(dict, tool))["function"]
    converted_tool = _format_dict_to_function_declaration(
        cast(FunctionDescription, function)
    )
    assert converted_tool.name == "People"
