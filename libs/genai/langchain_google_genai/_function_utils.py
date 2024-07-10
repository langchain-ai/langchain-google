from __future__ import annotations

from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
    cast,
)

import google.ai.generativelanguage as glm
from google.ai.generativelanguage import FunctionCallingConfig, FunctionDeclaration
from google.ai.generativelanguage import Tool as GoogleTool
from google.generativeai.types.content_types import ToolDict  # type: ignore[import]
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.json_schema import dereference_refs

TYPE_ENUM = {
    "string": glm.Type.STRING,
    "number": glm.Type.NUMBER,
    "integer": glm.Type.INTEGER,
    "boolean": glm.Type.BOOLEAN,
    "array": glm.Type.ARRAY,
    "object": glm.Type.OBJECT,
}

TYPE_ENUM_REVERSE = {v: k for k, v in TYPE_ENUM.items()}


class _ToolDictLike(TypedDict):
    function_declarations: _FunctionDeclarationLikeList


class _FunctionDeclarationDict(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Collection[str]]


class _ToolDict(TypedDict):
    function_declarations: Sequence[_FunctionDeclarationDict]


# Info: This is a FunctionDeclaration(=fc).
_FunctionDeclarationLike = Union[
    BaseTool, Type[BaseModel], FunctionDeclaration, Callable, Dict[str, Any]
]

# Info: This mean one tool.
_FunctionDeclarationLikeList = Sequence[_FunctionDeclarationLike]


# Info: This means one tool=Sequence of FunctionDeclaration
# The dict should be GoogleTool like. {"function_declarations": [ { "name": ...}.
# OpenAI like dict is not be accepted. {{'type': 'function', 'function': {'name': ...}
_ToolsType = Union[
    GoogleTool,
    ToolDict,
    _ToolDictLike,
    _FunctionDeclarationLikeList,
    _FunctionDeclarationLike,
]


#
# Info: GoogleTool means function_declarations and proto.Message.
def convert_to_genai_function_declarations(
    tool: _ToolsType,
) -> GoogleTool:
    if isinstance(tool, list):
        # multiple _FunctionDeclarationLike
        return GoogleTool(
            function_declarations=_convert_fc_likes_to_genai_function(tool)
        )
    elif isinstance(tool, (BaseTool, FunctionDeclaration)):
        # single _FunctionDeclarationLike
        return GoogleTool(
            function_declarations=[_convert_fc_like_to_genai_function(tool)]
        )
    elif isinstance(tool, type) and issubclass(tool, BaseModel):
        # single _FunctionDeclarationLike
        return GoogleTool(
            function_declarations=[_convert_fc_like_to_genai_function(tool)]
        )
    elif isinstance(tool, GoogleTool):
        return cast(GoogleTool, tool)
    elif callable(tool):
        return GoogleTool(
            function_declarations=[
                _convert_tool_to_genai_function(callable_as_lc_tool()(tool))
            ]
        )
    elif isinstance(tool, dict):
        return GoogleTool(function_declarations=_convert_dict_to_genai_functions(tool))  # type: ignore
    else:
        raise ValueError(f"Unsupported tool type {tool}")


def tool_to_dict(tool: GoogleTool) -> _ToolDict:
    function_declarations = []
    for function_declaration_proto in tool.function_declarations:
        properties: Dict[str, Any] = {}
        for property in function_declaration_proto.parameters.properties:
            property_type = function_declaration_proto.parameters.properties[
                property
            ].type
            property_dict = {"type": TYPE_ENUM_REVERSE[property_type]}
            property_description = function_declaration_proto.parameters.properties[
                property
            ].description
            if property_description:
                property_dict["description"] = property_description
            properties[property] = property_dict
        name = function_declaration_proto.name
        description = function_declaration_proto.description
        parameters = {"type": "object", "properties": properties}
        if function_declaration_proto.parameters.required:
            parameters["required"] = function_declaration_proto.parameters.required
        function_declaration = _FunctionDeclarationDict(
            name=name, description=description, parameters=parameters
        )
        function_declarations.append(function_declaration)
    return {"function_declarations": function_declarations}


def _convert_fc_likes_to_genai_function(
    fc_likes: _FunctionDeclarationLikeList,
) -> Sequence[FunctionDeclaration]:
    if isinstance(fc_likes, list):
        return [_convert_fc_like_to_genai_function(fc) for fc in fc_likes]
    raise ValueError(f"Unsupported fc_likes type {fc_likes}")


def _convert_fc_like_to_genai_function(
    fc_like: _FunctionDeclarationLike,
) -> FunctionDeclaration:
    if isinstance(fc_like, BaseTool):
        return _convert_tool_to_genai_function(fc_like)
    elif isinstance(fc_like, type) and issubclass(fc_like, BaseModel):
        return _convert_pydantic_to_genai_function(fc_like)
    elif isinstance(fc_like, dict):
        # TODO: add declaration_index
        return _convert_dict_to_genai_function(fc_like)
    elif callable(fc_like):
        return _convert_tool_to_genai_function(callable_as_lc_tool()(fc_like))
    else:
        raise ValueError(f"Unsupported fc_like type {fc_like}")


def _convert_tool_dict_to_genai_functions(
    tool_dict: _ToolDictLike,
) -> Sequence[FunctionDeclaration]:
    if "function_declarations" in tool_dict:
        return _convert_dicts_to_genai_functions(tool_dict["function_declarations"])  # type: ignore
    else:
        raise ValueError(f"Unsupported function tool_dict type {tool_dict}")


def _convert_dict_to_genai_functions(
    function_declarations_dict: Dict[str, Any],
) -> Sequence[FunctionDeclaration]:
    if "function_declarations" in function_declarations_dict:
        # GoogleTool like
        return [
            _convert_dict_to_genai_function(fc, i)
            for i, fc in enumerate(function_declarations_dict["function_declarations"])
        ]
    d = function_declarations_dict
    if "name" in d and "description" in d and "parameters" in d:
        # _FunctionDeclarationDict
        return [_convert_dict_to_genai_function(d)]
    else:
        # OpenAI like?
        raise ValueError(f"Unsupported function call type {function_declarations_dict}")


def _convert_dicts_to_genai_functions(
    function_declaration_dicts: Sequence[Dict[str, Any]],
) -> Sequence[FunctionDeclaration]:
    return [
        _convert_dict_to_genai_function(function_declaration_dict, i)
        for i, function_declaration_dict in enumerate(function_declaration_dicts)
    ]


def _convert_dict_to_genai_function(
    function_declaration_dict: Dict[str, Any], declaration_index: int = 0
) -> FunctionDeclaration:
    formatted_fc = {
        "name": function_declaration_dict.get("name", f"unknown-{declaration_index}"),
        "description": function_declaration_dict.get("description", "no-description"),
    }
    if "parameters" in function_declaration_dict:
        formatted_fc["parameters"] = {
            "properties": {
                k: {
                    "type_": TYPE_ENUM[v["type"]],
                    "description": v.get("description"),
                }
                for k, v in function_declaration_dict["parameters"][
                    "properties"
                ].items()
            },
            "required": function_declaration_dict.get("parameters", []).get(
                "required", []
            ),
            "type_": TYPE_ENUM[function_declaration_dict["parameters"]["type"]],
        }
    return FunctionDeclaration(**formatted_fc)


def _convert_tool_to_genai_function(tool: BaseTool) -> FunctionDeclaration:
    if tool.args_schema:
        fc = tool.args_schema
        if isinstance(fc, type) and issubclass(fc, BaseModel):
            return _convert_pydantic_to_genai_function(
                fc, tool_name=tool.name, tool_description=tool.description
            )
        raise ValueError(f"Unsupported function call type {fc}")
    else:
        return FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters={
                "properties": {
                    "__arg1": {"type_": TYPE_ENUM["string"]},
                },
                "required": ["__arg1"],
                "type_": TYPE_ENUM["object"],
            },
        )


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
) -> FunctionDeclaration:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)
    function_declaration = FunctionDeclaration(
        name=tool_name if tool_name else schema.get("title"),
        description=tool_description if tool_description else schema.get("description"),
        parameters={
            "properties": {
                k: {
                    "type_": _get_type_from_schema(v),
                    "description": v.get("description"),
                }
                for k, v in schema["properties"].items()
            },
            "required": schema.get("required", []),
            "type_": TYPE_ENUM[schema["type"]],
        },
    )
    return function_declaration


def _get_type_from_schema(schema: Dict[str, Any]) -> int:
    if "anyOf" in schema:
        types = [_get_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]]
        types = [t for t in types if t is not None]  # Remove None values
        if types:
            return types[-1]  # TODO: update FunctionDeclaration and pass all types?
        else:
            pass
    elif "type" in schema:
        stype = str(schema["type"])
        if stype in TYPE_ENUM:
            return TYPE_ENUM[stype]
        else:
            pass
    else:
        pass
    return TYPE_ENUM["string"]  # Default to string if no valid types found


_ToolChoiceType = Union[
    dict, List[str], str, Literal["auto", "none", "any"], Literal[True]
]


class _FunctionCallingConfigDict(TypedDict):
    mode: Union[FunctionCallingConfig.Mode, str]
    allowed_function_names: Optional[List[str]]


class _ToolConfigDict(TypedDict):
    function_calling_config: _FunctionCallingConfigDict


def _tool_choice_to_tool_config(
    tool_choice: _ToolChoiceType,
    all_names: List[str],
) -> _ToolConfigDict:
    allowed_function_names: Optional[List[str]] = None
    if tool_choice is True or tool_choice == "any":
        mode = "any"
        allowed_function_names = all_names
    elif tool_choice == "auto":
        mode = "auto"
    elif tool_choice == "none":
        mode = "none"
    elif isinstance(tool_choice, str):
        mode = "any"
        allowed_function_names = [tool_choice]
    elif isinstance(tool_choice, list):
        mode = "any"
        allowed_function_names = tool_choice
    elif isinstance(tool_choice, dict):
        if "mode" in tool_choice:
            mode = tool_choice["mode"]
            allowed_function_names = tool_choice.get("allowed_function_names")
        elif "function_calling_config" in tool_choice:
            mode = tool_choice["function_calling_config"]["mode"]
            allowed_function_names = tool_choice["function_calling_config"].get(
                "allowed_function_names"
            )
        else:
            raise ValueError(
                f"Unrecognized tool choice format:\n\n{tool_choice=}\n\nShould match "
                f"Google GenerativeAI ToolConfig or FunctionCallingConfig format."
            )
    else:
        raise ValueError(f"Unrecognized tool choice format:\n\n{tool_choice=}")
    return _ToolConfigDict(
        function_calling_config={
            "mode": mode,
            "allowed_function_names": allowed_function_names,
        }
    )
