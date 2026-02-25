from langchain_google_vertexai._anthropic_utils import _clean_content_block


def test_clean_content_block_removes_id_from_text() -> None:
    """Test that id is removed from text blocks (simulating tool_result issue)."""
    # This simulates the bad block from langchain-mcp-adapters
    bad_block = {"type": "text", "text": "some result", "id": "123-remove-me"}
    cleaned = _clean_content_block(bad_block)
    assert "id" not in cleaned
    assert cleaned["text"] == "some result"


def test_clean_content_block_preserves_id_in_tool_use() -> None:
    """Test that id is PRESERVED in tool_use blocks (required field)."""
    good_block = {
        "type": "tool_use",
        "name": "search",
        "input": {"query": "foo"},
        "id": "call_123",
    }
    cleaned = _clean_content_block(good_block)
    assert "id" in cleaned
    assert cleaned["id"] == "call_123"


def test_clean_content_block_preserves_id_in_image() -> None:
    """Test that id is PRESERVED in image blocks (required field)."""
    image_block = {
        "type": "image",
        "source": {"type": "base64", "data": "..."},
        "id": "img_123",
    }
    cleaned = _clean_content_block(image_block)
    assert "id" in cleaned
    assert cleaned["id"] == "img_123"
