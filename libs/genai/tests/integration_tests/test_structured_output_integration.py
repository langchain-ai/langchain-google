import json
import os

import pytest
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)

MODEL_NAME = "gemini-2.5-flash"


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    message: str
    count: int


class PersonResponse(BaseModel):
    """Person response model for testing."""

    name: str
    age: int
    skills: list[str]


class TextContent(BaseModel):
    """Text content for union testing."""

    type: str = "text"
    content: str


class NumberContent(BaseModel):
    """Number content for union testing."""

    type: str = "number"
    value: float


# Union type for testing anyOf support
ContentUnion = TextContent | NumberContent


class TreeNode(BaseModel):
    """Tree node for recursive schema testing."""

    value: str
    children: list["TreeNode"] | None = None


# Rebuild to resolve forward references
TreeNode.model_rebuild()


def test_basic_response_json_schema() -> None:
    """Test basic functionality."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["message", "count"],
    }

    llm_with_schema = llm.bind(
        response_mime_type="application/json", response_json_schema=schema
    )

    result = llm_with_schema.invoke("Respond with a message 'Hello World' and count 5")

    assert isinstance(result.content, str)
    response_data = json.loads(result.content)
    assert "message" in response_data
    assert "count" in response_data
    assert isinstance(response_data["message"], str)
    assert isinstance(response_data["count"], int)
    assert response_data["message"] == "Hello World"
    assert response_data["count"] == 5


def test_response_json_schema_union() -> None:
    """Test that `response_json_schema` works with unions"""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    schema_with_union = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "greeting"},
                    "message": {"type": "string"},
                },
                "required": ["type", "message"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "count"},
                    "number": {"type": "integer"},
                },
                "required": ["type", "number"],
            },
        ]
    }

    llm_schema = llm.bind(
        response_mime_type="application/json",
        response_json_schema=schema_with_union,
    )

    prompt = "Respond with type 'greeting' and message 'Hello there'"

    # Both should work with their respective schemas
    result = llm_schema.invoke(prompt)

    assert isinstance(result.content, str)
    response = json.loads(result.content)

    assert isinstance(response, dict)

    has_greeting = "type" in response and "message" in response
    has_count = "type" in response and "number" in response
    assert has_greeting or has_count, (
        f"Expected union schema structure, got: {response}"
    )


def test_json_schema_with_pydantic_model() -> None:
    """Test `json_schema` with a Pydantic model."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    structured_llm = llm.with_structured_output(SimpleResponse, method="json_schema")

    result = structured_llm.invoke(
        "Create a simple response with message 'Test successful' and count 42"
    )

    assert isinstance(result, SimpleResponse)
    assert result.message == "Test successful"
    assert result.count == 42


def test_json_schema_with_dict_schema() -> None:
    """Test `json_schema` with a `dict` schema."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["title", "priority"],
    }

    structured_llm = llm.with_structured_output(schema, method="json_schema")

    result = structured_llm.invoke(
        "Create a task with title 'Complete project' and priority 3"
    )

    assert isinstance(result, dict)
    assert "title" in result
    assert "priority" in result
    assert result["title"] == "Complete project"
    assert result["priority"] == 3


def test_recursive_schema_integration() -> None:
    """Test recursive schemas."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    structured_llm = llm.with_structured_output(TreeNode, method="json_schema")

    result = structured_llm.invoke(
        "Create a simple tree structure with root 'A' that has two children 'B' "
        "and 'C', where 'B' has one child 'D'"
    )

    assert isinstance(result, TreeNode)
    assert result.value == "A"
    assert result.children is not None
    assert len(result.children) == 2
    assert any(child.value == "B" for child in result.children)
    assert any(child.value == "C" for child in result.children)

    # Find the B node and check it has child D
    b_node = next(child for child in result.children if child.value == "B")
    assert b_node.children is not None
    assert len(b_node.children) == 1
    assert b_node.children[0].value == "D"


def test_union_schema_integration() -> None:
    """Test union schemas with `anyOf` support."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    # Test with union schema
    union_schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "text"},
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "number"},
                    "value": {"type": "number"},
                },
                "required": ["type", "value"],
            },
        ]
    }

    structured_llm = llm.with_structured_output(union_schema, method="json_schema")

    # Test with text response
    text_result = structured_llm.invoke(
        "Create a text content with type 'text' and the message 'Hello world'"
    )
    assert isinstance(text_result, dict)
    # Check that we got a valid response that matches one of the union options
    if "type" in text_result and "content" in text_result:
        # Text format response - verify it matches text schema
        assert isinstance(text_result["content"], str)
    elif "type" in text_result and "value" in text_result:
        # Number format response - verify it matches number schema
        assert isinstance(text_result["value"], (int, float))
    else:
        pytest.fail(f"Expected either text or number format, got: {text_result}")

    # Test with number response
    number_result = structured_llm.invoke(
        "Create a number content with type 'number' and value 42.5"
    )
    assert isinstance(number_result, dict)
    if "type" in number_result and "value" in number_result:
        # Number format response - verify it matches number schema
        assert isinstance(number_result["value"], (int, float))
    elif "type" in number_result and "content" in number_result:
        # Text format response - verify it matches text schema
        assert isinstance(number_result["content"], str)
    else:
        pytest.fail(f"Expected either text or number format, got: {number_result}")


def test_complex_schema_handling() -> None:
    """Test handling of complex schemas with constraints."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

    complex_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "score": {"type": "number", "minimum": 0, "maximum": 100},
                    },
                    "required": ["name", "score"],
                },
                "minItems": 2,
                "maxItems": 5,
            }
        },
        "required": ["items"],
    }

    structured_llm = llm.with_structured_output(complex_schema, method="json_schema")

    result = structured_llm.invoke(
        "Create a list with 3 items: Alice (score 95), Bob (score 87), Carol (score 92)"
    )

    assert isinstance(result, dict)
    assert "items" in result
    assert len(result["items"]) == 3

    for item in result["items"]:
        assert "name" in item
        assert "score" in item
        assert 0 <= item["score"] <= 100


def test_streaming_with_json_schema() -> None:
    """Test that streaming works with `json_schema`."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, streaming=True)

    structured_llm = llm.with_structured_output(SimpleResponse, method="json_schema")

    # Test streaming by collecting chunks
    chunks = []
    for chunk in structured_llm.stream(
        "Create response with message 'Streaming test' and count 7"
    ):
        chunks.append(chunk)  # noqa: PERF402

    # Should have received at least one chunk
    assert len(chunks) >= 1

    # Final result should be a SimpleResponse object
    final_result = chunks[-1]
    assert isinstance(final_result, SimpleResponse)
    assert final_result.message == "Streaming test"
    assert final_result.count == 7
