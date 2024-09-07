from typing import Any, Dict


def _simplify_anyof(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify 'anyOf' constructs in the schema containing nulls."""
    if "anyOf" in schema:
        anyof = schema["anyOf"]
        types = [subschema.get("type") for subschema in anyof]
        if "null" in types:
            # Remove 'null' type and simplify the schema
            non_null_schema = next(
                subschema for subschema in anyof if subschema.get("type") != "null"
            )
            schema = {**schema, **non_null_schema}
            schema.pop("anyOf")
    return schema


def transform_schema_v2_to_v1(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a Pydantic v2 schema to be more like a Pydantic v1 schema.
    """
    transformed_schema: Dict[str, Any] = {}
    required_fields = set(schema.get("required", []))

    for key, value in schema.items():
        if key == "properties":
            transformed_schema["properties"] = {}
            for prop_key, prop_value in value.items():
                simplified_prop = _simplify_anyof(prop_value)
                transformed_schema["properties"][prop_key] = transform_schema_v2_to_v1(
                    simplified_prop
                )
                if "anyOf" in prop_value and {"type": "null"} in prop_value["anyOf"]:
                    required_fields.discard(prop_key)
        elif key == "items":
            transformed_schema["items"] = transform_schema_v2_to_v1(value)
        elif key == "$defs":
            transformed_schema["definitions"] = {
                def_key: transform_schema_v2_to_v1(def_value)
                for def_key, def_value in value.items()
            }
        else:
            transformed_schema[key] = value

    if "required" in schema:
        transformed_schema["required"] = sorted(required_fields)

    return transformed_schema
