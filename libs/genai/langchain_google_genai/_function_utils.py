from __future__ import annotations

from typing import (
    Any,
    Callable,
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
from google.ai.generativelanguage import (
    FunctionCallingConfig,
    FunctionDeclaration,
)
from google.ai.generativelanguage import (
    Tool as GoogleTool,
)
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

_FunctionDeclarationLike = Union[
    BaseTool, Type[BaseModel], dict, Callable, FunctionDeclaration
]


class _ToolDict(TypedDict):
    function_declarations: Sequence[_FunctionDeclarationLike]


def convert_to_genai_function_declarations(
    tool: Union[
        GoogleTool,
        _ToolDict,
        _FunctionDeclarationLike,
        Sequence[_FunctionDeclarationLike],
    ],
) -> GoogleTool:
    if isinstance(tool, GoogleTool):
        return cast(GoogleTool, tool)
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        return GoogleTool(function_declarations=[_convert_to_genai_function(tool)])
    if callable(tool):
        return _convert_tool_to_genai_function(callable_as_lc_tool()(tool))
    if isinstance(tool, list):
        return convert_to_genai_function_declarations({"function_declarations": tool})
    if isinstance(tool, dict) and "function_declarations" in tool:
        return GoogleTool(
            function_declarations=[
                _convert_to_genai_function(fc) for fc in tool["function_declarations"]
            ],
        )
    return GoogleTool(function_declarations=[_convert_to_genai_function(tool)])  # type: ignore[arg-type]


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
        function_declaration = {
            "name": function_declaration_proto.name,
            "description": function_declaration_proto.description,
            "parameters": {"type": "object", "properties": properties},
        }
        if function_declaration_proto.parameters.required:
            function_declaration["parameters"][  # type: ignore[index]
                "required"
            ] = function_declaration_proto.parameters.required
        function_declarations.append(function_declaration)
    return {"function_declarations": function_declarations}


def _convert_to_genai_function(fc: _FunctionDeclarationLike) -> FunctionDeclaration:
    if isinstance(fc, BaseTool):
        return _convert_tool_to_genai_function(fc)
    elif isinstance(fc, type) and issubclass(fc, BaseModel):
        return _convert_pydantic_to_genai_function(fc)
    elif callable(fc):
        return _convert_tool_to_genai_function(callable_as_lc_tool()(fc))
    elif isinstance(fc, dict):
        formatted_fc = {"name": fc["name"], "description": fc.get("description")}
        if "parameters" in fc:
            formatted_fc["parameters"] = {
                "properties": {
                    k: {
                        "type_": TYPE_ENUM[v["type"]],
                        "description": v.get("description"),
                    }
                    for k, v in fc["parameters"]["properties"].items()
                },
                "required": fc.get("parameters", []).get("required", []),
                "type_": TYPE_ENUM[fc["parameters"]["type"]],
            }
        return FunctionDeclaration(**formatted_fc)
    else:
        raise ValueError(f"Unsupported function call type {fc}")


def _convert_tool_to_genai_function(tool: BaseTool) -> FunctionDeclaration:
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)
        return FunctionDeclaration(
            name=tool.name or schema["title"],
            description=tool.description or schema["description"],
            parameters={
                "properties": {
                    k: {
                        "type_": TYPE_ENUM[v["type"]],
                        "description": v.get("description"),
                    }
                    for k, v in schema["properties"].items()
                },
                "required": schema.get("required", []),
                "type_": TYPE_ENUM[schema["type"]],
            },
        )
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
) -> FunctionDeclaration:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)
    return FunctionDeclaration(
        name=schema["title"],
        description=schema.get("description", ""),
        parameters={
            "properties": {
                k: {
                    "type_": TYPE_ENUM[v["type"]],
                    "description": v.get("description"),
                }
                for k, v in schema["properties"].items()
            },
            "required": schema["required"],
            "type_": TYPE_ENUM[schema["type"]],
        },
    )


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
