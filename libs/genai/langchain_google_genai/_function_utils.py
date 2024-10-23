from __future__ import annotations

import collections
import importlib
import json
import logging
import os
import traceback
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
log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=log_level)

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
logger.debug("_ALLOWED_SCHEMA_FIELDS_SET=%s", _ALLOWED_SCHEMA_FIELDS_SET)


def is_pydantic_model(cls: Type) -> bool:
    # check attr for both Pydantic v1 and v2.
    return hasattr(cls, "__fields__") and isinstance(cls.__fields__, dict)


def get_pydantic_schema(model: Type[BaseModel]) -> Dict:
    try:
        # Pydantic v2
        return get_pydantic_schema_v2(model)
    except TypeError as e:
        logger.debug("TypeError get_pydantic_schema_v2 e=%s", e)

    try:
        # Pydantic v1
        return get_pydantic_schema_v1(model)
    except TypeError as e:
        logger.debug("TypeError get_pydantic_schema_v1 e=%s", e)

    return {}


def get_pydantic_schema_v2(model: Type[BaseModel]) -> Dict:
    if hasattr(model, "model_json_schema"):
        json_schema = model.model_json_schema()
        return json_schema

    raise TypeError(f"{model} is not Pydantic v2 model")


def get_pydantic_schema_v1(model: Type[BaseModel]) -> Dict:
    if hasattr(model, "schema_json"):
        schema_dict = model.schema()
        schema_json_str = json.dumps(schema_dict)
    else:
        raise TypeError(f"{model} is not Pydantic v1 model")
    schema_json = json.loads(schema_json_str)
    return schema_json


class _ToolDictLike(TypedDict):
    function_declarations: _FunctionDeclarationLikeList


class _FunctionDeclarationDict(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Collection[str]]


class _ToolDict(TypedDict):
    function_declarations: Sequence[_FunctionDeclarationDict]


# Info: Design overall.
# _function_utils.py has 2 steps.
#   step1: Prepare FunctionDeclaration(=gapic.Schema) when bind_tools() is called.
#   step2: Make request message when generate() is called.
#  both steps are used convert_to_genai_function_declarations()
#
# <<< step1 In/Out >>>
#   BaseTool ==> FunctionDeclaration
#     _format_base_tool_to_function_declaration()
#   pydantic(Type[BaseModel]) ==> FunctionDeclaration
#     _convert_pydantic_to_genai_function()
#   Dict(OpenAI tool-calling API) ==> FunctionDeclaration
#     _create_function_declaration_openai_dict()
#
# <<< step2 In/Out >>>
#   BaseTool ==> FunctionDeclaration
#     _format_base_tool_to_function_declaration()
#   pydantic(Type[BaseModel]) ==> FunctionDeclaration
#     _convert_pydantic_to_genai_function()
#   Dict(OpenAI tool-calling API) ==> FunctionDeclaration
#     _create_function_declaration_openai_dict()

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
        elif key in ["type"]:
            converted_schema["type_"] = _get_type_from_schema(schema)
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _dict_to_gapic_schema(schema: Dict[str, Any]) -> Optional[gapic.Schema]:
    logger.debug("_dict_to_gapic_schema\n  schema=%s", json.dumps(schema, indent=2))
    logger.debug("Stack trace:\n%s", "".join(traceback.format_stack()))
    if schema:
        dereferenced_schema = dereference_refs(schema)
        formatted_schema = _format_json_schema_to_gapic(dereferenced_schema)
        json_schema = json.dumps(formatted_schema)
        logger.debug(
            "_dict_to_gapic_schema\n  json_schema=%s", json.dumps(json_schema, indent=2)
        )
        return gapic.Schema.from_json(json_schema)
    return None


def _format_dict_to_function_declaration(
    tool: Union[FunctionDescription, Dict[str, Any]],
) -> gapic.FunctionDeclaration:
    print(tool)
    return gapic.FunctionDeclaration(
        name=tool.get("name") or tool.get("title"),
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
                logger.debug("fd=", json.dumps(fd, indent=2))
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
        logger.debug("_format_to_gapic_function_declaration BaseTool")
        return _format_base_tool_to_function_declaration(tool)
    elif isinstance(tool, type) and is_basemodel_subclass_safe(tool):
        logger.debug("_format_to_gapic_function_declaration BaseModel")
        return _convert_pydantic_to_genai_function(tool)  # type: ignore[arg-type]
    elif isinstance(tool, dict):
        if all(k in tool for k in ("name", "description")) and "parameters" not in tool:
            logger.debug("_format_to_gapic_function_declaration dict(no parameters1)")
            function = cast(dict, tool)
            function["parameters"] = {}
        else:
            if (
                "parameters" in tool and tool["parameters"].get("properties")  # type: ignore[index]
            ):
                logger.debug("_format_to_gapic_function_declaration dict(via openai)")
                function = convert_to_openai_tool(cast(dict, tool))["function"]
            else:
                logger.debug(
                    "_format_to_gapic_function_declaration dict(no parameters2)"
                )
                function = cast(dict, tool)
                function["parameters"] = {}
        return _format_dict_to_function_declaration(cast(FunctionDescription, function))
    elif callable(tool):
        return _format_base_tool_to_function_declaration(callable_as_lc_tool()(tool))
    raise ValueError(f"Unsupported tool type {tool}")


def _create_function_declaration_openai_dict(tool: dict) -> Dict:
    function = cast(dict, tool)
    if all(k in tool for k in ("name", "description")) and "parameters" not in tool:
        function["parameters"] = {}
    else:
        if "parameters" in tool and tool["parameters"].get("properties"):
            function = convert_to_openai_tool(cast(dict, tool))["function"]
        else:
            function = cast(dict, tool)
            function["parameters"] = {}
    return function


def _format_base_tool_to_function_declaration(
    tool: BaseTool,
) -> gapic.FunctionDeclaration:
    if not tool.args_schema:
        parameters = _create_function_declaration_parameters({})
        gapic_parameters = _dict_to_gapic_schema(parameters)
        return gapic.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=gapic_parameters,
        )

    if issubclass(tool.args_schema, BaseModel):
        schema = tool.args_schema.model_json_schema()
    elif issubclass(tool.args_schema, BaseModelV1):
        schema = tool.args_schema.schema()
    else:
        raise NotImplementedError(
            f"args_schema must be a Pydantic BaseModel, got {tool.args_schema}."
        )
    gapic_parameters = _dict_to_gapic_schema(schema)

    return gapic.FunctionDeclaration(
        name=tool.name or schema.get("title"),
        description=tool.description or schema.get("description"),
        parameters=gapic_parameters,
    )


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
) -> gapic.FunctionDeclaration:
    schema = get_pydantic_schema(pydantic_model)
    schema = dereference_refs(schema)
    schema.pop("definitions", None)

    name = tool_name if tool_name else schema.get("title")
    description = tool_description if tool_description else schema.get("description")
    function_declaration = _create_function_declaration(
        name=str(name), description=str(description), schema=schema
    )

    return function_declaration


def _create_function_declaration(
    name: str, description: str, schema: Dict[str, Any]
) -> gapic.FunctionDeclaration:
    logger.debug("_convert_pydantic_to_genai_function\n  schema=%s", schema)
    parameters = _create_function_declaration_parameters(schema)
    gapic_parameters = _dict_to_gapic_schema(parameters)
    logger.debug(
        "_convert_pydantic_to_genai_function\n  gapic_parameters=%s", gapic_parameters
    )
    function_declaration = gapic.FunctionDeclaration(
        name=name if name else schema.get("title"),
        description=description if description else schema.get("description"),
        parameters=parameters,
    )
    return function_declaration


def _get_function_declaration_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_function_declaration_from_schema(schema)
    return {}


def _get_function_declaration_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling?hl#functiondeclaration
      name: str
      description: str
      parameters: Schema
      response: Schema(not support)
    """
    function_declaration = {}
    for k, v in schema.items():
        # name
        name = v.get("name")
        if name:
            function_declaration["name"] = name

        # description
        description = v.get("description")
        if description:
            function_declaration["description"] = description

        # parameters
        parameters = v.get("parameters")
        if parameters:
            function_declaration["parameters"] = (
                _create_function_declaration_parameters(parameters)
            )

    return function_declaration


def _create_function_declaration_parameters(schema: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(
        "_create_function_declaration_parameters schema=\n%s",
        json.dumps(schema, indent=2),
    )
    parameters = {
        # "properties": _get_schema_schema_from_dict(schema.get("properties")),
        "properties": _get_properties_from_schema_any(schema.get("properties")),
        # "items": _get_items_from_schema_any(
        #     schema
        # ),  # TODO: fix it https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling?hl#schema
        "required": schema.get("required", []),
        "type_": _get_type_from_schema(schema),
    }
    logger.info(
        "_create_function_declaration_parameters parameters=\n%s",
        json.dumps(parameters, indent=2),
    )
    return parameters


def _get_properties_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_properties_from_schema(schema)
    return {}


def _get_schema_schema_from_dict(schema_dict: Dict) -> Dict[str, Any]:
    """
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling?hl#schema
      type: str
      enum: str[]
      items	: Schema[]
      properties: Schema
      required: str[]
      nullable: bool(not support)
    """
    logger.debug(
        "_get_schema_schema_from_dict\n  schema_dict=%s",
        json.dumps(schema_dict, indent=2),
    )

    # type(mandatory)
    schema_schema: Dict[str, Any] = {}
    ttype = _get_type_from_schema(schema_dict)
    schema_schema["type_"] = ttype

    # TODO: add loop for each item like properties
    for k, v in schema_dict.items():
        # enum
        if k == "enum":
            schema_schema["enum"] = v
            continue

        # items
        if k == "items" and ttype == TYPE_ENUM["array"]:
            schema_schema["items"] = _get_items_from_schema_any(v)
            continue

        # properties
        if k == "properties" and ttype == TYPE_ENUM["object"]:
            schema_schema["properties"] = _get_schema_schema_from_dict(v)
            continue

        # required
        if k == "required":
            schema_schema["required"] = v
            continue

        # other allowed is pass through
        if k in _ALLOWED_SCHEMA_FIELDS_SET:
            if k != "type":
                schema_schema[k] = v

    logger.debug("_get_schema_schema_from_dict\n  schema_schema=%s", schema_schema)
    return schema_schema


def _get_properties_from_schema(schema: Dict) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    # this loop is each property. {val_name: Schema}
    for k, v in schema.items():
        logger.debug("_get_properties_from_schema\n  %s=%s", k, v)
        if not isinstance(k, str):
            logger.warning(f"Key '{k}' is not supported in schema, type={type(k)}")
            continue
        if not isinstance(v, Dict):
            logger.warning(f"Value '{v}' is not supported in schema, ignoring v={v}")
            continue
        properties_item: Dict[str, Union[str, int, Dict, List]] = {}

        # type(mandatory)
        ttype = _get_type_from_schema(v)
        properties_item["type_"] = ttype

        # enum
        enum_val = v.get("enum")
        if enum_val:
            properties_item["enum"] = enum_val

        # description
        description = v.get("description")
        if description and isinstance(description, str):
            properties_item["description"] = description

        # items
        if ttype == TYPE_ENUM["array"] and v.get("items"):
            properties_item["items"] = _get_items_from_schema_any(v.get("items"))

        # properties
        if ttype == TYPE_ENUM["object"] and v.get("properties"):
            properties_item["properties"] = _get_properties_from_schema_any(
                v.get("properties")
            )

        # optional(custom description)
        if k == "title" and "description" not in properties_item:
            properties_item["description"] = k + " is " + str(v)

        properties[k] = properties_item

    return properties


def _get_items_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_items_from_schema(schema)
    if isinstance(schema, List):
        return _get_items_from_schema(schema)
    if isinstance(schema, str):
        return _get_items_from_schema(schema)
    return {}


def _get_items_from_schema(schema: Union[Dict, List, str]) -> Dict[str, Any]:
    items: Dict = {}
    if isinstance(schema, List):
        for i, v in enumerate(schema):
            items[f"item{i}"] = _get_items_from_schema(v)
    elif isinstance(schema, Dict):
        item = _get_item_from_schema(schema)
        items = item
    else:
        # str
        items["type_"] = TYPE_ENUM.get(str(schema), glm.Type.STRING)
    return items


def _get_item_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    item = _get_schema_schema_from_dict(schema)
    return item


def _get_type_from_schema(schema: Union[str, Dict[str, Any]]) -> int:
    if isinstance(schema, str):
        return TYPE_ENUM["string"]
    elif "anyOf" in schema:
        types = [_get_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]]
        types = [t for t in types if t is not None]  # Remove None values
        if types:
            if "string" in types:
                # support string case only.
                return TYPE_ENUM["string"]
            return types[0]  # TODO: return all types and complete schema each type.
        else:
            pass
    elif "type" in schema:
        stype = str(schema["type"])
        return TYPE_ENUM.get(stype, glm.Type.STRING)
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


def is_basemodel_subclass_safe(tool: Type) -> bool:
    if safe_import("langchain_core.utils.pydantic", "is_basemodel_subclass"):
        from langchain_core.utils.pydantic import (
            is_basemodel_subclass,  # type: ignore[import]
        )

        return is_basemodel_subclass(tool)
    else:
        return issubclass(tool, BaseModel)


def safe_import(module_name: str, attribute_name: str = "") -> bool:
    try:
        module = importlib.import_module(module_name)
        if attribute_name:
            return hasattr(module, attribute_name)
        return True
    except ImportError:
        return False
