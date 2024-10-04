from typing import Any, List

import pytest

from langchain_google_vertexai._utils import (
    GoogleModelFamily,
    _get_def_key_from_schema_path,
    replace_defs_in_schema,
)


@pytest.mark.parametrize(
    "srcs,exp",
    [
        (
            [
                "chat-bison@001",
                "text-bison@002",
                "medlm-medium",
                "medlm-large",
            ],
            GoogleModelFamily.PALM,
        ),
        (
            [
                "code-bison@002",
                "code-gecko@002",
            ],
            GoogleModelFamily.CODEY,
        ),
        (
            [
                "gemini-1.0-pro-001",
                "gemini-1.0-pro-002",
                "gemini-1.0-pro-vision-001",
                "gemini-1.0-pro-vision",
                "medlm-medium@latest",
            ],
            GoogleModelFamily.GEMINI,
        ),
        (
            [
                "gemini-1.5-flash-preview-0514",
                "gemini-1.5-pro-preview-0514",
                "gemini-1.5-pro-preview-0409",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-001",
                "medlm-large-1.5-preview",
                "medlm-large-1.5-001",
            ],
            GoogleModelFamily.GEMINI_ADVANCED,
        ),
    ],
)
def test_google_model_family(srcs: List[str], exp: GoogleModelFamily):
    for src in srcs:
        res = GoogleModelFamily(src)
        assert res == exp


def test_valid_schema_path():
    schema_path = "#/$defs/MyDefinition"
    expected_key = "MyDefinition"
    assert _get_def_key_from_schema_path(schema_path) == expected_key


@pytest.mark.parametrize(
    "schema_path",
    [123, "#/definitions/MyDefinition", "#/$defs/MyDefinition/extra", "#/$defs"],
)
def test_invalid_schema_path(schema_path: Any):
    with pytest.raises(ValueError):
        _get_def_key_from_schema_path(schema_path)


def test_schema_no_defs():
    schema = {"type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_empty_defs():
    schema = {"$defs": {}, "type": "integer"}
    expected_schema = {"type": "integer"}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_simple_ref_replacement():
    schema = {
        "$defs": {"MyDefinition": {"type": "string"}},
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {"property": {"type": "string"}}
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_nested_ref_replacement():
    schema = {
        "$defs": {
            "MyDefinition": {
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/NameDefinition"}},
            },
            "NameDefinition": {"type": "string"},
        },
        "property": {"$ref": "#/$defs/MyDefinition"},
    }
    expected_schema = {
        "property": {"type": "object", "properties": {"name": {"type": "string"}}}
    }
    assert replace_defs_in_schema(schema) == expected_schema


def test_schema_recursive_error_self_reference():
    schema = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }
        },
        "root": {"$ref": "#/$defs/Node"},
    }
    with pytest.raises(RecursionError):
        _ = replace_defs_in_schema(schema)
