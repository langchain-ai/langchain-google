from __future__ import annotations

from typing import (
    List,
    Sequence,
    Type,
    Union,
)

import google.ai.generativelanguage as glm
from google.generativeai.types import Tool as GoogleTool  # type: ignore[import]
from google.generativeai.types.content_types import (  # type: ignore[import]
    FunctionDeclarationType,
    ToolDict,
    ToolType,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.json_schema import dereference_refs

TYPE_ENUM = {
    "string": glm.Type.STRING,
    "number": glm.Type.NUMBER,
    "integer": glm.Type.INTEGER,
    "boolean": glm.Type.BOOLEAN,
    "array": glm.Type.ARRAY,
    "object": glm.Type.OBJECT,
}


def convert_to_genai_function_declarations(
    tool: Union[GoogleTool, ToolDict, Sequence[FunctionDeclarationType]],
) -> List[ToolType]:
    if isinstance(tool, GoogleTool):
        return [tool]
    # check whether a dict is supported by glm, otherwise we parse it explicitly
    if isinstance(tool, dict):
        first_function_declaration = tool.get("function_declarations", [None])[0]
        if isinstance(first_function_declaration, glm.FunctionDeclaration):
            return [tool]
        schema = None
        try:
            schema = first_function_declaration.parameters
        except AttributeError:
            pass
        if schema is None:
            schema = first_function_declaration.get("parameters")
        if schema is None or isinstance(schema, glm.Schema):
            return [tool]
        return [
            glm.Tool(
                function_declarations=[_convert_to_genai_function(fc)],
            )
            for fc in tool["function_declarations"]
        ]
    return [
        glm.Tool(
            function_declarations=[_convert_to_genai_function(fc)],
        )
        for fc in tool
    ]


def _convert_to_genai_function(fc: FunctionDeclarationType) -> glm.FunctionDeclaration:
    if isinstance(fc, BaseTool):
        print(fc)
        return _convert_tool_to_genai_function(fc)
    elif isinstance(fc, type) and issubclass(fc, BaseModel):
        return _convert_pydantic_to_genai_function(fc)
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
