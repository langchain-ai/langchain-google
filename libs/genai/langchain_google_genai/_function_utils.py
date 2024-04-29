from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
)

import google.ai.generativelanguage as glm
from google.generativeai.types import Tool as GoogleTool  # type: ignore[import]
from google.generativeai.types.content_types import (  # type: ignore[import]
    FunctionCallingConfigType,
    FunctionDeclarationType,
    ToolDict,
    ToolType,
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


def convert_to_genai_function_declarations(
    tool: Union[
        GoogleTool, ToolDict, FunctionDeclarationType, Sequence[FunctionDeclarationType]
    ],
) -> ToolType:
    """Convert any tool-like object to a ToolType.

    https://github.com/google-gemini/generative-ai-python/blob/668695ebe3e9de496a36eeb95cb2ed2faba9b939/google/generativeai/types/content_types.py#L574
    """
    if isinstance(tool, GoogleTool):
        return tool
    # check whether a dict is supported by glm, otherwise we parse it explicitly
    if isinstance(tool, dict):
        first_function_declaration = tool.get("function_declarations", [None])[0]
        if isinstance(first_function_declaration, glm.FunctionDeclaration):
            return tool
        schema = None
        try:
            schema = first_function_declaration.parameters
        except AttributeError:
            pass
        if schema is None:
            schema = first_function_declaration.get("parameters")
        if schema is None or isinstance(schema, glm.Schema):
            return tool
        return glm.Tool(
            function_declarations=[
                _convert_to_genai_function(fc) for fc in tool["function_declarations"]
            ],
        )
    elif isinstance(tool, type) and issubclass(tool, BaseModel):
        return glm.Tool(function_declarations=[_convert_to_genai_function(tool)])
    elif callable(tool):
        return _convert_tool_to_genai_function(callable_as_lc_tool()(tool))
    elif isinstance(tool, list):
        return glm.Tool(
            function_declarations=[_convert_to_genai_function(fc) for fc in tool]
        )
    return glm.Tool(function_declarations=[_convert_to_genai_function(tool)])


def tool_to_dict(tool: Union[glm.Tool, GoogleTool]) -> ToolDict:
    if isinstance(tool, GoogleTool):
        tool = tool._proto
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


def _convert_to_genai_function(fc: FunctionDeclarationType) -> glm.FunctionDeclaration:
    if isinstance(fc, BaseTool):
        return _convert_tool_to_genai_function(fc)
    elif isinstance(fc, type) and issubclass(fc, BaseModel):
        return _convert_pydantic_to_genai_function(fc)
    elif callable(fc):
        return _convert_tool_to_genai_function(callable_as_lc_tool()(fc))
    elif isinstance(fc, dict):
        return glm.FunctionDeclaration(
            name=fc["name"],
            description=fc.get("description"),
            parameters={
                "properties": {
                    k: {
                        "type_": TYPE_ENUM[v["type"]],
                        "description": v.get("description"),
                    }
                    for k, v in fc["parameters"]["properties"].items()
                },
                "required": fc["parameters"].get("required", []),
                "type_": TYPE_ENUM[fc["parameters"]["type"]],
            },
        )
    else:
        raise ValueError(f"Unsupported function call type {fc}")


def _convert_tool_to_genai_function(tool: BaseTool) -> glm.FunctionDeclaration:
    if tool.args_schema:
        schema = dereference_refs(tool.args_schema.schema())
        schema.pop("definitions", None)
        return glm.FunctionDeclaration(
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
        return glm.FunctionDeclaration(
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
) -> glm.FunctionDeclaration:
    schema = dereference_refs(pydantic_model.schema())
    schema.pop("definitions", None)
    return glm.FunctionDeclaration(
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


class _ToolConfigDict(TypedDict):
    function_calling_config: FunctionCallingConfigType


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
