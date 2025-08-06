from __future__ import annotations

import collections
import importlib
import logging
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

from google.genai import types
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    convert_to_openai_tool,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


TYPE_ENUM = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "array": types.Type.ARRAY,
    "object": types.Type.OBJECT,
    "null": None,
}

# Note: For google.genai, we'll use a simplified approach for allowed schema fields
# since the new library doesn't expose protobuf fields in the same way
_ALLOWED_SCHEMA_FIELDS = [
    "type",
    "type_",
    "description",
    "enum",
    "format",
    "items",
    "properties",
    "required",
    "nullable",
    "anyOf",
    "default",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "title",
]
_ALLOWED_SCHEMA_FIELDS_SET = set(_ALLOWED_SCHEMA_FIELDS)


# Info: This is a FunctionDeclaration(=fc).
_FunctionDeclarationLike = Union[
    BaseTool, Type[BaseModel], types.FunctionDeclaration, Callable, Dict[str, Any]
]
_GoogleSearchRetrievalLike = Union[
    types.GoogleSearchRetrieval,
    Dict[str, Any],
]
_GoogleSearchLike = Union[types.GoogleSearch, Dict[str, Any]]
_CodeExecutionLike = Union[types.ToolCodeExecution, Dict[str, Any]]


class _ToolDict(TypedDict):
    function_declarations: Sequence[_FunctionDeclarationLike]
    google_search_retrieval: Optional[_GoogleSearchRetrievalLike]
    google_search: NotRequired[_GoogleSearchLike]
    code_execution: NotRequired[_CodeExecutionLike]


# Info: This means one tool=Sequence of FunctionDeclaration
# The dict should be Tool like. {"function_declarations": [ { "name": ...}.
# OpenAI like dict is not be accepted. {{'type': 'function', 'function': {'name': ...}
_ToolType = Union[types.Tool, _ToolDict, _FunctionDeclarationLike]
_ToolsType = Sequence[_ToolType]


def _format_json_schema_to_gapic(schema: Dict[str, Any]) -> Dict[str, Any]:
    converted_schema: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "definitions":
            continue
        elif key == "items":
            if value is not None:
                converted_schema["items"] = _format_json_schema_to_gapic(value)
        elif key == "properties":
            converted_schema["properties"] = _get_properties_from_schema(value)
            continue
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic(value[0])
        elif key in ["type", "type_"]:
            if isinstance(value, dict):
                converted_schema["type"] = value["_value_"]
            elif isinstance(value, str):
                converted_schema["type"] = value
            else:
                raise ValueError(f"Invalid type: {value}")
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _dict_to_genai_schema(schema: Dict[str, Any]) -> Optional[types.Schema]:
    if schema:
        dereferenced_schema = dereference_refs(schema)
        formatted_schema = _format_json_schema_to_gapic(dereferenced_schema)

        # Convert the formatted schema to google.genai.types.Schema
        schema_dict = {}
        if "type" in formatted_schema:
            type_value = "STRING"
            type_obj = formatted_schema["type"]
            if isinstance(type_obj, dict):
                type_value = type_obj["_value_"]
            elif isinstance(type_obj, str):
                type_value = type_obj
            else:
                raise ValueError(f"Invalid type: {type_obj}")
            schema_dict["type"] = types.Type(type_value)
        if "description" in formatted_schema:
            schema_dict["description"] = formatted_schema["description"]
        if "title" in formatted_schema:
            schema_dict["title"] = formatted_schema["title"]
        if "properties" in formatted_schema:
            schema_dict["properties"] = formatted_schema["properties"]
        # Always set required to empty list if not present
        schema_dict["required"] = formatted_schema.get("required", [])
        if "items" in formatted_schema:
            schema_dict["items"] = formatted_schema["items"]
        if "enum" in formatted_schema:
            schema_dict["enum"] = formatted_schema["enum"]
        if "nullable" in formatted_schema:
            schema_dict["nullable"] = formatted_schema["nullable"]

        return types.Schema.model_validate(schema_dict)
    return None


def _format_dict_to_function_declaration(
    tool: Union[FunctionDescription, Dict[str, Any]],
) -> types.FunctionDeclaration:
    name = tool.get("name") or tool.get("title") or "MISSING_NAME"
    description = tool.get("description") or None
    parameters = _dict_to_genai_schema(tool.get("parameters", {}))
    return types.FunctionDeclaration(
        name=str(name),
        description=description,
        parameters=parameters,
    )


# Info: Tool means function_declarations and other tool types.
def convert_to_genai_function_declarations(
    tools: _ToolsType,
) -> types.Tool:
    if not isinstance(tools, collections.abc.Sequence):
        logger.warning(
            "convert_to_genai_function_declarations expects a Sequence "
            "and not a single tool."
        )
        tools = [tools]

    tool_dict: Dict[str, Any] = {}
    function_declarations: List[types.FunctionDeclaration] = []

    for tool in tools:
        if isinstance(tool, types.Tool):
            # Handle existing Tool objects
            if hasattr(tool, "function_declarations") and tool.function_declarations:
                function_declarations.extend(tool.function_declarations)
            if (
                hasattr(tool, "google_search_retrieval")
                and tool.google_search_retrieval
            ):
                if "google_search_retrieval" in tool_dict:
                    raise ValueError(
                        "Providing multiple google_search_retrieval"
                        " or mixing with function_declarations is not supported"
                    )
                tool_dict["google_search_retrieval"] = tool.google_search_retrieval
            if hasattr(tool, "google_search") and tool.google_search:
                tool_dict["google_search"] = tool.google_search
            if hasattr(tool, "code_execution") and tool.code_execution:
                tool_dict["code_execution"] = tool.code_execution
        elif isinstance(tool, dict):
            # not _ToolDictLike
            if not any(
                f in tool
                for f in [
                    "function_declarations",
                    "google_search_retrieval",
                    "google_search",
                    "code_execution",
                ]
            ):
                fd = _format_to_genai_function_declaration(tool)  # type: ignore[arg-type]
                function_declarations.append(fd)
                continue
            # _ToolDictLike
            tool = cast(_ToolDict, tool)
            if "function_declarations" in tool:
                tool_function_declarations = tool["function_declarations"]
                if tool_function_declarations is not None and not isinstance(
                    tool["function_declarations"], collections.abc.Sequence
                ):
                    raise ValueError(
                        "function_declarations should be a list"
                        f"got '{type(tool_function_declarations)}'"
                    )
                if tool_function_declarations:
                    fds = [
                        _format_to_genai_function_declaration(fd)
                        for fd in tool_function_declarations
                    ]
                    function_declarations.extend(fds)
            if "google_search_retrieval" in tool:
                if "google_search_retrieval" in tool_dict:
                    raise ValueError(
                        "Providing multiple google_search_retrieval"
                        " or mixing with function_declarations is not supported"
                    )
                if isinstance(tool["google_search_retrieval"], dict):
                    tool_dict["google_search_retrieval"] = types.GoogleSearchRetrieval(
                        **tool["google_search_retrieval"]
                    )
                else:
                    tool_dict["google_search_retrieval"] = tool[
                        "google_search_retrieval"
                    ]
            if "google_search" in tool:
                if isinstance(tool["google_search"], dict):
                    tool_dict["google_search"] = types.GoogleSearch(
                        **tool["google_search"]
                    )
                else:
                    tool_dict["google_search"] = tool["google_search"]
            if "code_execution" in tool:
                if isinstance(tool["code_execution"], dict):
                    tool_dict["code_execution"] = types.ToolCodeExecution(
                        **tool["code_execution"]
                    )
                else:
                    tool_dict["code_execution"] = tool["code_execution"]
        else:
            fd = _format_to_genai_function_declaration(tool)  # type: ignore[arg-type]
            function_declarations.append(fd)

    if function_declarations:
        tool_dict["function_declarations"] = function_declarations

    return types.Tool(**tool_dict)


def tool_to_dict(tool: types.Tool) -> _ToolDict:
    def _traverse_values(raw: Any) -> Any:
        if isinstance(raw, list):
            return [_traverse_values(v) for v in raw]
        if isinstance(raw, dict):
            processed = {k: _traverse_values(v) for k, v in raw.items()}
            return processed
        if hasattr(raw, "__dict__"):
            return _traverse_values(raw.__dict__)
        return raw

    if hasattr(tool, "model_dump"):
        raw_result = tool.model_dump()
    else:
        raw_result = tool.__dict__

    result = _traverse_values(raw_result)
    return result


def _format_to_genai_function_declaration(
    tool: _FunctionDeclarationLike,
) -> types.FunctionDeclaration:
    if isinstance(tool, BaseTool):
        return _format_base_tool_to_function_declaration(tool)
    elif isinstance(tool, type) and is_basemodel_subclass_safe(tool):
        return _convert_pydantic_to_genai_function(tool)
    elif isinstance(tool, dict):
        if all(k in tool for k in ("type", "function")) and tool["type"] == "function":
            function = tool["function"]
        elif (
            all(k in tool for k in ("name", "description")) and "parameters" not in tool
        ):
            function = cast(dict, tool)
        else:
            if (
                "parameters" in tool and tool["parameters"].get("properties")  # type: ignore[index]
            ):
                function = convert_to_openai_tool(cast(dict, tool))["function"]
            else:
                function = cast(dict, tool)
        function["parameters"] = function.get("parameters") or {}
        # Empty 'properties' field not supported.
        if not function["parameters"].get("properties"):
            function["parameters"] = {}
        return _format_dict_to_function_declaration(cast(FunctionDescription, function))
    elif callable(tool):
        return _format_base_tool_to_function_declaration(callable_as_lc_tool()(tool))
    raise ValueError(f"Unsupported tool type {tool}")


def _format_base_tool_to_function_declaration(
    tool: BaseTool,
) -> types.FunctionDeclaration:
    if not tool.args_schema:
        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "__arg1": types.Schema(type=types.Type.STRING),
                },
                required=["__arg1"],
            ),
        )

    if isinstance(tool.args_schema, dict):
        schema = tool.args_schema
    elif issubclass(tool.args_schema, BaseModel):
        schema = tool.args_schema.model_json_schema()
    elif issubclass(tool.args_schema, BaseModelV1):
        schema = tool.args_schema.schema()
    else:
        raise NotImplementedError(
            "args_schema must be a Pydantic BaseModel or JSON schema, "
            f"got {tool.args_schema}."
        )
    parameters = _dict_to_genai_schema(schema)

    return types.FunctionDeclaration(
        name=tool.name or schema.get("title"),
        description=tool.description or schema.get("description"),
        parameters=parameters,
    )


def _convert_pydantic_to_genai_function(
    pydantic_model: Type[BaseModel],
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
) -> types.FunctionDeclaration:
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

    # Convert to google.genai Schema format
    parameters_dict = {
        "type": TYPE_ENUM[schema["type"]],
        "properties": _get_properties_from_schema_any(schema.get("properties")),
        "required": schema.get("required", []),
    }

    function_declaration = types.FunctionDeclaration(
        name=tool_name if tool_name else schema.get("title"),
        description=tool_description if tool_description else schema.get("description"),
        parameters=types.Schema(**parameters_dict),
    )
    return function_declaration


def _get_properties_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_properties_from_schema(schema)
    return {}


def _get_properties_from_schema(schema: Dict) -> Dict[str, Any]:
    properties: Dict[str, Dict[str, Union[str, int, Dict, List]]] = {}
    for k, v in schema.items():
        if not isinstance(k, str):
            logger.warning(f"Key '{k}' is not supported in schema, type={type(k)}")
            continue
        if not isinstance(v, Dict):
            logger.warning(f"Value '{v}' is not supported in schema, ignoring v={v}")
            continue
        properties_item: Dict[str, Union[str, int, Dict, List]] = {}
        if v.get("anyOf") and all(
            anyOf_type.get("type") != "null" for anyOf_type in v.get("anyOf", [])
        ):
            properties_item["anyOf"] = [
                _format_json_schema_to_gapic(anyOf_type)
                for anyOf_type in v.get("anyOf", [])
            ]
        elif v.get("type") or v.get("anyOf") or v.get("type_"):
            item_type_ = _get_type_from_schema(v)
            properties_item["type"] = item_type_
            if _is_nullable_schema(v):
                properties_item["nullable"] = True

            # Replace `v` with chosen definition for array / object json types
            any_of_types = v.get("anyOf")
            if any_of_types and item_type_ in [types.Type.ARRAY, types.Type.OBJECT]:
                json_type_ = "array" if item_type_ == types.Type.ARRAY else "object"
                # Use Index -1 for consistency with `_get_nullable_type_from_schema`
                v = [val for val in any_of_types if val.get("type") == json_type_][-1]

        if v.get("enum"):
            properties_item["enum"] = v["enum"]

        description = v.get("description")
        if description and isinstance(description, str):
            properties_item["description"] = description

        if properties_item.get("type") == types.Type.ARRAY and v.get("items"):
            properties_item["items"] = _get_items_from_schema_any(v.get("items"))

        if properties_item.get("type") == types.Type.OBJECT:
            if (
                v.get("anyOf")
                and isinstance(v["anyOf"], list)
                and isinstance(v["anyOf"][0], dict)
            ):
                v = v["anyOf"][0]
            v_properties = v.get("properties")
            if v_properties:
                properties_item["properties"] = _get_properties_from_schema_any(
                    v_properties
                )
                if isinstance(v_properties, dict):
                    properties_item["required"] = [
                        k for k, v in v_properties.items() if "default" not in v
                    ]
            else:
                # Providing dummy type for object without properties
                properties_item["type"] = types.Type.STRING

        if k == "title" and "description" not in properties_item:
            properties_item["description"] = k + " is " + str(v)

        properties[k] = properties_item

    return properties


def _get_items_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, (dict, list, str)):
        return _get_items_from_schema(schema)
    return {}


def _get_items_from_schema(schema: Union[Dict, List, str]) -> Dict[str, Any]:
    items: Dict = {}
    if isinstance(schema, List):
        for i, v in enumerate(schema):
            items[f"item{i}"] = _get_properties_from_schema_any(v)
    elif isinstance(schema, Dict):
        items["type"] = _get_type_from_schema(schema)
        if items["type"] == types.Type.OBJECT and "properties" in schema:
            items["properties"] = _get_properties_from_schema_any(schema["properties"])
        if items["type"] == types.Type.ARRAY and "items" in schema:
            items["items"] = _format_json_schema_to_gapic(schema["items"])
        if "title" in schema or "description" in schema:
            items["description"] = schema.get("description") or schema.get("title")
        if _is_nullable_schema(schema):
            items["nullable"] = True
        if "required" in schema:
            items["required"] = schema["required"]
    else:
        # str
        items["type"] = _get_type_from_schema({"type": schema})
        if _is_nullable_schema({"type": schema}):
            items["nullable"] = True

    return items


def _get_type_from_schema(schema: Dict[str, Any]) -> types.Type:
    type_ = _get_nullable_type_from_schema(schema)
    return type_ if type_ is not None else types.Type.STRING


def _get_nullable_type_from_schema(schema: Dict[str, Any]) -> Optional[types.Type]:
    if "anyOf" in schema:
        schema_types = [
            _get_nullable_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]
        ]
        schema_types = [t for t in schema_types if t is not None]  # Remove None values
        # TODO: update FunctionDeclaration and pass all types?
        if schema_types:
            return schema_types[-1]
        else:
            pass
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, types.Type):
            return type_
        elif isinstance(type_, int):
            raise ValueError(f"Invalid type, int not supported: {type_}")
        elif isinstance(type_, dict):
            return types.Type(type_["_value_"])
        elif isinstance(type_, str):
            if type_ == "null":
                return None
            return types.Type(type_)
        else:
            return None
    else:
        pass
    return None  # Default to string if no valid types found


def _is_nullable_schema(schema: Dict[str, Any]) -> bool:
    if "anyOf" in schema:
        schema_types = [
            _get_nullable_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]
        ]
        return any(t is None for t in schema_types)
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, types.Type):
            return False
        elif isinstance(type_, int):
            # Handle integer type values (from tool_to_dict serialization)
            # Integer types are never null (except for NULL type handled separately)
            return type_ == 7  # 7 corresponds to NULL type
    else:
        pass
    return False


_ToolChoiceType = Union[
    dict, List[str], str, Literal["auto", "none", "any"], Literal[True]
]


def _tool_choice_to_tool_config(
    tool_choice: _ToolChoiceType,
    all_names: List[str],
) -> types.ToolConfig:
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
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode(mode),
            allowed_function_names=allowed_function_names,
        )
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


def replace_defs_in_schema(original_schema: dict, defs: Optional[dict] = None) -> dict:
    """Given an OpenAPI schema with a property '$defs' replaces all occurrences of
    referenced items in the dictionary.

    Args:
        original_schema: Schema generated by `BaseModel.model_schema_json`
        defs: Definitions for recursive calls.

    Returns:
        Schema with refs replaced.
    """

    new_defs = defs or original_schema.get("$defs")

    if new_defs is None or not isinstance(new_defs, dict):
        return original_schema.copy()

    resulting_schema = {}

    for key, value in original_schema.items():
        if key == "$defs":
            continue

        if not isinstance(value, dict):
            resulting_schema[key] = value
        else:
            if "$ref" in value:
                new_value = value.copy()

                path = new_value.pop("$ref")
                def_key = _get_def_key_from_schema_path(path)
                new_item = new_defs.get(def_key)

                assert isinstance(new_item, dict)
                new_value.update(new_item)

                resulting_schema[key] = replace_defs_in_schema(new_value, defs=new_defs)
            else:
                resulting_schema[key] = replace_defs_in_schema(value, defs=new_defs)

    return resulting_schema


def _get_def_key_from_schema_path(schema_path: str) -> str:
    error_message = f"Malformed schema reference path {schema_path}"

    if not isinstance(schema_path, str) or not schema_path.startswith("#/$defs/"):
        raise ValueError(error_message)

    # Schema has to have only one extra level.
    parts = schema_path.split("/")
    if len(parts) != 3:
        raise ValueError(error_message)

    return parts[-1]


# Backward compatibility alias
_dict_to_gapic_schema = _dict_to_genai_schema
