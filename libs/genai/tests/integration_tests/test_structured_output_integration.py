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


def test_basic_response_json_schema(backend_config: dict) -> None:
    """Test basic functionality."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["message", "count"],
    }

    llm_with_schema = llm.bind(
        response_mime_type="application/json", response_json_schema=schema
    )

    result = llm_with_schema.invoke("Respond with a message 'Hello World' and count 5")

    if isinstance(result.content, list) and len(result.content) > 0:
        # Extract text from structured content block
        content_block = result.content[0]
        if isinstance(content_block, dict) and "text" in content_block:
            json_text = content_block["text"]
        else:
            json_text = str(content_block)
    else:
        json_text = str(result.content)

    response_data = json.loads(json_text)
    assert "message" in response_data
    assert "count" in response_data
    assert isinstance(response_data["message"], str)
    assert isinstance(response_data["count"], int)
    assert response_data["message"] == "Hello World"
    assert response_data["count"] == 5


def test_response_json_schema_union(backend_config: dict) -> None:
    """Test that `response_json_schema` works with unions"""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

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

    if isinstance(result.content, list) and len(result.content) > 0:
        # Extract text from structured content block
        content_block = result.content[0]
        if isinstance(content_block, dict) and "text" in content_block:
            json_text = content_block["text"]
        else:
            json_text = str(content_block)
    else:
        json_text = str(result.content)

    response = json.loads(json_text)

    assert isinstance(response, dict)

    has_greeting = "type" in response and "message" in response
    has_count = "type" in response and "number" in response
    assert has_greeting or has_count, (
        f"Expected union schema structure, got: {response}"
    )


def test_json_schema_with_pydantic_model(backend_config: dict) -> None:
    """Test `json_schema` with a Pydantic model."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)
    structured_llm = llm.with_structured_output(SimpleResponse, method="json_schema")

    result = structured_llm.invoke(
        "Create a simple response with message 'Test successful' and count 42"
    )

    assert isinstance(result, SimpleResponse)
    assert result.message == "Test successful"
    assert result.count == 42


def test_json_schema_with_dict_schema(backend_config: dict) -> None:
    """Test `json_schema` with a `dict` schema."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

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


def test_recursive_schema_integration(backend_config: dict) -> None:
    """Test recursive schemas."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

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


def test_union_schema_integration(backend_config: dict) -> None:
    """Test union schemas with `anyOf` support."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

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


def test_complex_schema_handling(backend_config: dict) -> None:
    """Test handling of complex schemas with constraints."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

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


def test_streaming_with_json_schema(backend_config: dict) -> None:
    """Test that streaming works with `json_schema`."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, streaming=True, **backend_config)

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


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_streaming_with_json_schema_output_versions(
    output_version: str, backend_config: dict
) -> None:
    """Test that streaming works with `json_schema` for different output versions."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        streaming=True,
        output_version=output_version,
        **backend_config,
    )

    structured_llm = llm.with_structured_output(SimpleResponse, method="json_schema")

    # Test streaming by collecting chunks
    chunks = []
    for chunk in structured_llm.stream(
        "Create response with message 'Version test' and count 42"
    ):
        chunks.append(chunk)  # noqa: PERF402

    # Should have received at least one chunk
    assert len(chunks) >= 1

    # Final result should be a SimpleResponse object
    final_result = chunks[-1]
    assert isinstance(final_result, SimpleResponse)
    assert final_result.message == "Version test"
    assert final_result.count == 42


async def test_async_streaming_with_json_schema(backend_config: dict) -> None:
    """Test that async streaming works with `json_schema`."""
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, streaming=True, **backend_config)

    structured_llm = llm.with_structured_output(SimpleResponse, method="json_schema")

    # Test async streaming by collecting chunks
    chunks = []
    async for chunk in structured_llm.astream(
        "Create response with message 'Async streaming test' and count 99"
    ):
        chunks.append(chunk)  # noqa: PERF401

    # Should have received at least one chunk
    assert len(chunks) >= 1

    # Final result should be a SimpleResponse object
    final_result = chunks[-1]
    assert isinstance(final_result, SimpleResponse)
    assert final_result.message == "Async streaming test"
    assert final_result.count == 99


def test_streaming_raw_chunks_accumulation(backend_config: dict) -> None:
    """Test that raw `AIMessageChunk` objects can be accumulated before parsing.

    At the LLM level (before the `PydanticOutputParser`), we get `AIMessageChunk`
    objects that can be accumulated using the + operator. This test verifies
    that accumulation works correctly at the raw level.
    """
    from langchain_core.messages import AIMessageChunk

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, streaming=True, **backend_config)

    # Bind JSON schema at LLM level (without parser)
    llm_with_schema = llm.bind(
        response_mime_type="application/json",
        response_json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age", "skills"],
        },
    )

    # Collect and accumulate raw chunks
    chunks = []
    accumulated: AIMessageChunk | None = None
    for chunk in llm_with_schema.stream(
        "Create a person with name 'Bob Smith', age 25, "
        "and skills: ['Java', 'Kubernetes']"
    ):
        chunks.append(chunk)
        if accumulated is None:
            accumulated = chunk  # type: ignore[assignment]
        else:
            accumulated = accumulated + chunk  # type: ignore[assignment]

    # Should have received chunks
    assert len(chunks) >= 1

    # All chunks should be AIMessageChunk objects
    for chunk in chunks:
        assert isinstance(chunk, AIMessageChunk)

    # Accumulated result should be a valid AIMessageChunk
    assert isinstance(accumulated, AIMessageChunk)
    assert accumulated.content  # Should have content

    # Extract JSON from content (could be string or list of content blocks)
    content = accumulated.content
    if isinstance(content, list) and len(content) > 0:
        # Extract text from first content block
        first_block = content[0]
        if isinstance(first_block, dict) and "text" in first_block:
            json_str = first_block["text"]
        elif isinstance(first_block, str):
            json_str = first_block
        else:
            json_str = str(first_block)
    else:
        json_str = str(content)

    # The content should be valid JSON
    result_json = json.loads(json_str)
    assert "name" in result_json
    assert "age" in result_json
    assert "skills" in result_json
    assert result_json["name"] == "Bob Smith"
    assert result_json["age"] == 25


def test_streaming_parsed_output_behavior(backend_config: dict) -> None:
    """Test streaming behavior with PydanticOutputParser.

    When using `with_structured_output` with `json_schema` method,
    the PydanticOutputParser buffers the raw chunks and emits
    fully-parsed Pydantic objects. This test verifies that behavior.
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, streaming=True, **backend_config)

    structured_llm = llm.with_structured_output(PersonResponse, method="json_schema")

    # Collect all chunks from the parser
    chunks: list[PersonResponse] = []
    for chunk in structured_llm.stream(
        "Create a person with name 'Alice Johnson', age 30, "
        "and skills: ['Python', 'Machine Learning', 'Data Science']"
    ):
        chunks.append(chunk)  # type: ignore[arg-type]  # noqa: PERF402

    # Parser emits complete Pydantic objects (not incremental JSON strings)
    assert len(chunks) >= 1

    # All chunks should be complete PersonResponse objects
    for chunk in chunks:
        assert isinstance(chunk, PersonResponse)
        # Each chunk should have valid, complete data
        assert chunk.name
        assert chunk.age > 0
        assert len(chunk.skills) > 0

    # Verify the final result has all expected data
    final_chunk = chunks[-1]
    assert final_chunk.name == "Alice Johnson"
    assert final_chunk.age == 30
    assert len(final_chunk.skills) == 3
    assert "Python" in final_chunk.skills
    assert "Machine Learning" in final_chunk.skills
    assert "Data Science" in final_chunk.skills


def test_moderation_union_schema(backend_config: dict) -> None:
    """Test Union types work correctly."""

    class SpamDetails(BaseModel):
        """Details for content classified as spam."""

        reason: str
        spam_type: str

    class NotSpamDetails(BaseModel):
        """Details for content classified as not spam."""

        summary: str
        is_safe: bool

    class ModerationResult(BaseModel):
        """The result of content moderation."""

        decision: SpamDetails | NotSpamDetails

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, **backend_config)

    # Test passing Pydantic model directly
    structured_llm_model = llm.with_structured_output(
        ModerationResult, method="json_schema"
    )

    # Test passing schema dict directly
    structured_llm_dict = llm.with_structured_output(
        ModerationResult.model_json_schema(), method="json_schema"
    )

    safe_prompt = "Review this content: 'Great recipe for chocolate cake! Thanks!'"
    spam_prompt = "Review this content: 'Click here to win $1000000!!!'"

    # Test with Pydantic model (should return Pydantic object)
    safe_result_model = structured_llm_model.invoke(safe_prompt)
    assert isinstance(safe_result_model, ModerationResult)
    assert hasattr(safe_result_model.decision, "summary") or hasattr(
        safe_result_model.decision, "reason"
    )

    # Test with dict schema (should return dict)
    safe_result_dict = structured_llm_dict.invoke(safe_prompt)
    assert isinstance(safe_result_dict, dict)
    assert "decision" in safe_result_dict

    # Test spam detection
    spam_result = structured_llm_model.invoke(spam_prompt)
    assert isinstance(spam_result, ModerationResult)
