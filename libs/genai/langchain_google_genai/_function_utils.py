from __future__ import annotations

import collections
import json
import logging
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
import google.ai.generativelanguage_v1beta.types as gapic
import proto  # type: ignore[import]
from google.generativeai.types.content_types import ToolDict  # type: ignore[import]
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    convert_to_openai_tool,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

logger = logging.getLogger(__name__)


TYPE_ENUM = {
    "string": glm.Type.STRING,
    "number": glm.Type.NUMBER,
    "integer": glm.Type.INTEGER,
    "boolean": glm.Type.BOOLEAN,
    "array": glm.Type.ARRAY,
    "object": glm.Type.OBJECT,
}

TYPE_ENUM_REVERSE = {v: k for k, v in TYPE_ENUM.items()}
_ALLOWED_SCHEMA_FIELDS = []
_ALLOWED_SCHEMA_FIELDS.extend([f.name for f in gapic.Schema()._pb.DESCRIPTOR.fields])
_ALLOWED_SCHEMA_FIELDS.extend(
    [
        f
        for f in gapic.Schema.to_dict(
            gapic.Schema(), preserving_proto_field_name=False
        ).keys()
    ]
)
_ALLOWED_SCHEMA_FIELDS_SET = set(_ALLOWED_SCHEMA_FIELDS)


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
    BaseTool, Type[BaseModel], gapic.FunctionDeclaration, Callable, Dict[str, Any]
]

# Info: This mean one tool.
_FunctionDeclarationLikeList = Sequence[_FunctionDeclarationLike]


# Info: This means one tool=Sequence of FunctionDeclaration
# The dict should be gapic.Tool like. {"function_declarations": [ { "name": ...}.
# OpenAI like dict is not be accepted. {{'type': 'function', 'function': {'name': ...}
_ToolsType = Union[
    gapic.Tool,
    ToolDict,
    _ToolDictLike,
    _FunctionDeclarationLikeList,
    _FunctionDeclarationLike,
]


def _format_json_schema_to_gapic(schema: Dict[str, Any]) -> Dict[str, Any]:
    converted_schema: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "definitions":
            continue
        elif key == "items":
            converted_schema["items"] = _format_json_schema_to_gapic(value)
        elif key == "properties":
            if "properties" not in converted_schema:
                converted_schema["properties"] = {}
            for pkey, pvalue in value.items():
                converted_schema["properties"][pkey] = _format_json_schema_to_gapic(
                    pvalue
                )
            continue
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic(value[0])
        elif key in ["type", "_type"]:
            converted_schema["type"] = str(value).upper()
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _dict_to_gapic_schema(schema: Dict[str, Any]) -> Optional[gapic.Schema]:
    if schema:
        dereferenced_schema = dereference_refs(schema)
        formatted_schema = _format_json_schema_to_gapic(dereferenced_schema)
        json_schema = json.dumps(formatted_schema)
        return gapic.Schema.from_json(json_schema)
    return None


def _format_dict_to_function_declaration(
    tool: Union[FunctionDescription, Dict[str, Any]],
) -> gapic.FunctionDeclaration:
    return gapic.FunctionDeclaration(
        name=tool.get("name"),
        description=tool.get("description"),
        parameters=_dict_to_gapic_schema(tool.get("parameters", {})),
    )


# Info: gapic.Tool means function_declarations and proto.Message.
def convert_to_genai_function_declarations(
    tools: Sequence[_ToolsType],
) -> gapic.Tool:
    if not isinstance(tools, collections.abc.Sequence):
        logger.warning(
            "convert_to_genai_function_declarations expects a Sequence "
            "and not a single tool."
        )
        tools = [tools]
    gapic_tool = gapic.Tool()
    for tool in tools:
        if isinstance(tool, gapic.Tool):
            gapic_tool.function_declarations.extend(tool.function_declarations)
        elif isinstance(tool, dict):
            if "function_declarations" not in tool:
                fd = _format_to_gapic_function_declaration(tool)
                gapic_tool.function_declarations.append(fd)
                continue
            tool = cast(_ToolDictLike, tool)
            function_declarations = tool["function_declarations"]
            if not isinstance(function_declarations, collections.abc.Sequence):
                raise ValueError(
                    "function_declarations should be a list"
                    f"got '{type(function_declarations)}'"
                )
            if function_declarations:
                fds = [
                    _format_to_gapic_function_declaration(fd)
                    for fd in function_declarations
                ]
                gapic_tool.function_declarations.extend(fds)
        else:
            fd = _format_to_gapic_function_declaration(tool)
            gapic_tool.function_declarations.append(fd)
    return gapic_tool


def tool_to_dict(tool: gapic.Tool) -> _ToolDict:
    def _traverse_values(raw: Any) -> Any:
        if isinstance(raw, list):
            return [_traverse_values(v) for v in raw]
        if isinstance(raw, dict):
            return {k: _traverse_values(v) for k, v in raw.items()}
        if isinstance(raw, proto.Message):
            return _traverse_values(type(raw).to_dict(raw))
        return raw

    return _traverse_values(type(tool).to_dict(tool))


def _format_to_gapic_function_declaration(
    tool: _FunctionDeclarationLike,
) -> gapic.FunctionDeclaration:
    if isinstance(tool, BaseTool):
        return _format_base_tool_to_function_declaration(tool)
    elif isinstance(tool, type) and issubclass(tool, BaseModel):
        return _convert_pydantic_to_genai_function(tool)
    elif isinstance(tool, dict):
        if all(k in tool for k in ("name", "description")) and "parameters" not in tool:
            function = cast(dict, tool)
            function["parameters"] = {}
        else:
            if "parameters" in tool and tool["parameters"].get("properties"):  # type: ignore[index]
                function = convert_to_openai_tool(cast(dict, tool))["function"]
            else:
                function = cast(dict, tool)
                function["parameters"] = {}
        return _format_dict_to_function_declaration(cast(FunctionDescription, function))
    elif callable(tool):
        return _format_base_tool_to_function_declaration(callable_as_lc_tool()(tool))
    raise ValueError(f"Unsupported tool type {tool}")


def _format_base_tool_to_function_declaration(
    tool: BaseTool,
) -> gapic.FunctionDeclaration:
    if not tool.args_schema:
        return gapic.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=gapic.Schema(
                type=gapic.Type.OBJECT,
                properties={
                    "__arg1": gapic.Schema(type=gapic.Type.STRING),
                },
                required=["__arg1"],
            ),
        )

    if issubclass(tool.args_schema, BaseModel):
        schema = tool.args_schema.model_json_schema()
    elif issubclass(tool.args_schema, BaseModelV1):
        schema = tool.args_schema.schema()
    else:
        raise NotImplementedError(
            f"args_schema must be a Pydantic BaseModel, got {tool.args_schema}."
        )
    parameters = _dict_to_gapic_schema(schema)

    return gapic.FunctionDeclaration(
        name=tool.name or schema.get("title"),
        description=tool.description or schema.get("description"),
        parameters=parameters,
    )


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
) -> gapic.FunctionDeclaration:
    if issubclass(pydantic_model, BaseModel):
        schema = pydantic_model.model_json_schema()
    elif issubclass(pydantic_model, BaseModelV1):
        schema = pydantic_model.schema()
    else:
        raise NotImplementedError(
            f"pydantic_model must be a Pydantic BaseModel, got {pydantic_model}"
        )
    schema = dereference_refs(schema)
    schema.pop("definitions", None)
    function_declaration = gapic.FunctionDeclaration(
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
    mode: Union[gapic.FunctionCallingConfig.Mode, str]
    allowed_function_names: Optional[List[str]]


class _ToolConfigDict(TypedDict):
    function_calling_config: _FunctionCallingConfigDict


def _tool_choice_to_tool_config(
    tool_choice: _ToolChoiceType,
    all_names: List[str],
) -> _ToolConfigDict:
    allowed_function_names: Optional[List[str]] = None
    if tool_choice is True or tool_choice == "any":
        mode = "ANY"
        allowed_function_names = all_names
    elif tool_choice == "auto":
        mode = "AUTO"
    elif tool_choice == "none":
        mode = "NONE"
    elif isinstance(tool_choice, str):
        mode = "ANY"
        allowed_function_names = [tool_choice]
    elif isinstance(tool_choice, list):
        mode = "ANY"
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
            "mode": mode.upper(),
            "allowed_function_names": allowed_function_names,
        }
    )
