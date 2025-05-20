from __future__ import annotations

import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import google.ai.generativelanguage as glm
import google.cloud.aiplatform_v1beta1.types as gapic
from langchain_core.utils.json_schema import dereference_refs

logger = logging.getLogger(__name__)

TYPE_ENUM = {
    "string": glm.Type.STRING,
    "number": glm.Type.NUMBER,
    "integer": glm.Type.INTEGER,
    "boolean": glm.Type.BOOLEAN,
    "array": glm.Type.ARRAY,
    "object": glm.Type.OBJECT,
    "null": None,
}


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


def dict_to_gapic_json_schema(
    schema: Dict[str, Any], pydantic_version: str = "v1"
) -> str:
    # Resolve refs in schema because $refs and $defs are not supported
    # by the Gemini API.
    dereferenced_schema = dereference_refs(schema)

    if pydantic_version == "v1":
        formatted_schema = _format_json_schema_to_gapic_v1(dereferenced_schema)
    else:
        formatted_schema = _format_json_schema_to_gapic(dereferenced_schema)

    return json.dumps(formatted_schema)


def _format_json_schema_to_gapic_v1(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Format a JSON schema from a Pydantic V1 BaseModel to gapic."""
    converted_schema: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "definitions":
            continue
        elif key == "items":
            converted_schema["items"] = _format_json_schema_to_gapic_v1(value)
        elif key == "properties":
            converted_schema["properties"] = _get_properties_from_schema(value)
            continue
        elif key in ["type", "_type"]:
            converted_schema["type"] = str(value).upper()
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic_v1(value[0])
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _format_json_schema_to_gapic(
    schema: Dict[str, Any],
    parent_key: Optional[str] = None,
    required_fields: Optional[list] = None,
) -> Dict[str, Any]:
    """Format a JSON schema from a Pydantic V2 BaseModel to gapic."""
    converted_schema: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "$defs":
            continue
        elif key == "items":
            converted_schema["items"] = _format_json_schema_to_gapic(
                value, parent_key, required_fields
            )
        elif key == "properties":
            if "properties" not in converted_schema:
                converted_schema["properties"] = {}
            for pkey, pvalue in value.items():
                converted_schema["properties"][pkey] = _format_json_schema_to_gapic(
                    pvalue, pkey, schema.get("required", [])
                )
            continue
        elif key in ["type", "_type"]:
            converted_schema["type"] = str(value).upper()
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic(value[0], parent_key, required_fields)
        elif key == "anyOf":
            if len(value) == 2 and any(v.get("type") == "null" for v in value):
                non_null_type = next(v for v in value if v.get("type") != "null")
                converted_schema.update(
                    _format_json_schema_to_gapic(
                        non_null_type, parent_key, required_fields
                    )
                )
                # Remove the field from required if it exists
                if required_fields and parent_key in required_fields:
                    required_fields.remove(parent_key)
                continue
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


# Get Properties from Schema
def _get_properties_from_schema_any(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Dict):
        return _get_properties_from_schema(schema)
    return {}


def _get_properties_from_schema(schema: Dict) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    for k, v in schema.items():
        if not isinstance(k, str):
            logger.warning(f"Key '{k}' is not supported in schema, type={type(k)}")
            continue
        if not isinstance(v, Dict):
            logger.warning(f"Value '{v}' is not supported in schema, ignoring v={v}")
            continue
        properties_item: Dict[str, Union[str, int, Dict, List]] = {}
        if v.get("type") or v.get("anyOf") or v.get("type_"):
            item_type_ = _get_type_from_schema(v)
            properties_item["type_"] = item_type_
            if _is_nullable_schema(v):
                properties_item["nullable"] = True

            # Replace `v` with chosen definition for array / object json types
            any_of_types = v.get("anyOf")
            if any_of_types and item_type_ in [glm.Type.ARRAY, glm.Type.OBJECT]:
                json_type_ = "array" if item_type_ == glm.Type.ARRAY else "object"
                # Use Index -1 for consistency with `_get_nullable_type_from_schema`
                v = [val for val in any_of_types if val.get("type") == json_type_][-1]

        if v.get("enum"):
            properties_item["enum"] = v["enum"]

        v_title = v.get("title")
        if v_title and isinstance(v_title, str):
            properties_item["title"] = v_title

        description = v.get("description")
        if description and isinstance(description, str):
            properties_item["description"] = description

        if properties_item.get("type_") == glm.Type.ARRAY and v.get("items"):
            properties_item["items"] = _get_items_from_schema_any(v.get("items"))

        if properties_item.get("type_") == glm.Type.OBJECT:
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
                properties_item["type_"] = glm.Type.STRING

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
        items["type_"] = _get_type_from_schema(schema)
        if items["type_"] == glm.Type.OBJECT and "properties" in schema:
            items["properties"] = _get_properties_from_schema_any(schema["properties"])
        if items["type_"] == glm.Type.ARRAY and "items" in schema:
            items["items"] = _format_json_schema_to_gapic_v1(schema["items"])
        if "title" in schema or "description" in schema:
            items["description"] = (
                schema.get("description") or schema.get("title") or ""
            )
        if _is_nullable_schema(schema):
            items["nullable"] = True
        if "required" in schema:
            items["required"] = schema["required"]
    else:
        # str
        items["type_"] = _get_type_from_schema({"type": schema})
        if _is_nullable_schema({"type": schema}):
            items["nullable"] = True

    return items


def _get_type_from_schema(schema: Dict[str, Any]) -> int:
    return _get_nullable_type_from_schema(schema) or glm.Type.STRING


def _get_nullable_type_from_schema(schema: Dict[str, Any]) -> Optional[int]:
    if "anyOf" in schema:
        types = [
            _get_nullable_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]
        ]
        types = [t for t in types if t is not None]  # Remove None values
        if types:
            return types[-1]  # TODO: update FunctionDeclaration and pass all types?
        else:
            pass
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, int):
            return type_
        stype = str(schema["type"]) if "type" in schema else str(schema["type_"])
        return TYPE_ENUM.get(stype, glm.Type.STRING)
    else:
        pass
    return glm.Type.STRING  # Default to string if no valid types found


def _is_nullable_schema(schema: Dict[str, Any]) -> bool:
    if "anyOf" in schema:
        types = [
            _get_nullable_type_from_schema(sub_schema) for sub_schema in schema["anyOf"]
        ]
        return any(t is None for t in types)
    elif "type" in schema or "type_" in schema:
        type_ = schema["type"] if "type" in schema else schema["type_"]
        if isinstance(type_, int):
            return False
        stype = str(schema["type"]) if "type" in schema else str(schema["type_"])
        return TYPE_ENUM.get(stype, glm.Type.STRING) is None
    else:
        pass
    return False
