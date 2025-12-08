"""Unit tests for _anthropic_utils.py."""

import base64
from unittest.mock import patch

import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    SignatureDelta,
    ThinkingDelta,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import create_image_block, create_text_block
from langchain_core.messages.tool import tool_call as create_tool_call

from langchain_google_vertexai._anthropic_utils import (
    _documents_in_params,
    _format_image,
    _format_message_anthropic,
    _format_messages_anthropic,
    _make_message_chunk_from_anthropic_event,
    _thinking_in_params,
)


def test_format_message_anthropic_with_cache_control_in_kwargs() -> None:
    """Test formatting a message with cache control in additional_kwargs."""
    message = HumanMessage(
        content="Hello", additional_kwargs={"cache_control": {"type": "semantic"}}
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ],
    }


def test_format_message_anthropic_with_cache_control_in_block() -> None:
    """Test formatting a message with cache control in content block."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ]
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ],
    }


def test_format_message_anthropic_with_mixed_blocks() -> None:
    """Test formatting a message with mixed blocks, some with cache control."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}},
            {"type": "text", "text": "World"},
            "Plain text",
        ]
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}},
            {"type": "text", "text": "World"},
            {"type": "text", "text": "Plain text"},
        ],
    }


def test_format_messages_anthropic_with_system_cache_control() -> None:
    """Test formatting messages with system message having cache control."""
    messages = [
        SystemMessage(
            content="System message",
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        ),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(
        messages, project="test-project"
    )

    assert system_messages == [
        {
            "type": "text",
            "text": "System message",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    assert formatted_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]


def test_format_message_anthropic_system() -> None:
    """Test formatting a system message."""
    message = SystemMessage(
        content="System message",
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == [
        {
            "type": "text",
            "text": "System message",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_format_message_anthropic_system_list() -> None:
    """Test formatting a system message with list content."""
    message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": "System rule 1",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "System rule 2"},
        ]
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == [
        {
            "type": "text",
            "text": "System rule 1",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "System rule 2"},
    ]


def test_format_message_anthropic_with_chain_of_thoughts() -> None:
    """Test formatting a system message with chain of thoughts."""
    message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": "final output of the model",
            },
            {
                "type": "thinking",
                "thinking": "thoughts of the model...",
                "signature": "thinking-signature",
                "additional_keys": "additional_values",
            },
            {
                "type": "redacted_thinking",
                "data": "redacted-thoughts-data",
                "additional_keys": "additional_values",
            },
        ]
    )
    result = _format_message_anthropic(message, project="test-project")
    assert result == [
        {
            "type": "text",
            "text": "final output of the model",
        },
        {
            "type": "thinking",
            "thinking": "thoughts of the model...",
            "signature": "thinking-signature",
        },
        {"type": "redacted_thinking", "data": "redacted-thoughts-data"},
    ]


def test_format_messages_anthropic_with_system_string() -> None:
    """Test formatting messages with system message as string."""
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(
        messages, project="test-project"
    )

    assert system_messages == [{"type": "text", "text": "System message"}]

    assert formatted_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]


def test_format_messages_anthropic_with_system_list() -> None:
    """Test formatting messages with system message as a list."""
    messages = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "System rule 1",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "System rule 2"},
            ]
        ),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(
        messages, project="test-project"
    )

    assert system_messages == [
        {
            "type": "text",
            "text": "System rule 1",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "System rule 2"},
    ]

    assert formatted_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]


def test_format_messages_anthropic_with_system_mixed_list() -> None:
    """Test formatting messages with system message as a mixed list."""
    messages = [
        SystemMessage(
            content=[
                "Plain system rule",
                {
                    "type": "text",
                    "text": "Formatted system rule",
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        ),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(
        messages, project="test-project"
    )

    assert system_messages == [
        {"type": "text", "text": "Plain system rule"},
        {
            "type": "text",
            "text": "Formatted system rule",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    assert formatted_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]


def test_format_messages_anthropic_with_mixed_messages() -> None:
    """Test formatting a conversation with various message types and cache controls."""
    messages = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "System message",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Human message",
                    "cache_control": {"type": "semantic"},
                }
            ]
        ),
        AIMessage(
            content="AI response",
            additional_kwargs={"cache_control": {"type": "semantic"}},
        ),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(
        messages, project="test-project"
    )

    assert system_messages == [
        {
            "type": "text",
            "text": "System message",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    assert formatted_messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Human message",
                    "cache_control": {"type": "semantic"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "AI response",
                    "cache_control": {"type": "semantic"},
                }
            ],
        },
    ]


@pytest.mark.parametrize(
    ("source_history", "expected_sm", "expected_history"),
    [
        (
            [
                AIMessage(
                    content="",
                ),
            ],
            None,
            [],
        ),
        (
            [
                AIMessage(
                    content=[],
                ),
            ],
            None,
            [],
        ),
        (
            [
                AIMessage(
                    content=[""],
                ),
            ],
            None,
            [],
        ),
        (
            [
                AIMessage(
                    content=["", "Mike age is 30"],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Mike age is 30",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content="Mike age is 30",
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Mike age is 30",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=[{"type": "text", "text": "Mike age is 30"}],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Mike age is 30",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=[{"type": "text", "text": ""}],
                ),
            ],
            None,
            [],
        ),
        (
            [
                SystemMessage(content="test1"),
                AIMessage(
                    content="Mike age is 30",
                ),
            ],
            [{"type": "text", "text": "test1"}],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Mike age is 30",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30", "Arthur age is 30"],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Mike age is 30"},
                        {"type": "text", "text": "Arthur age is 30"},
                    ],
                }
            ],
        ),
        (
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,/9j/4AAQSk"},
                        }
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "/9j/4AAQSk",
                            },
                        },
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=[],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        }
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content="Mike age is 30",
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Mike age is 30"},
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        },
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30", "Arthur age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Mike age is 30"},
                        {"type": "text", "text": "Arthur age is 30"},
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        },
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Rob"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
                AIMessage(
                    content=["Arthur age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Mike age is 30"},
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Rob"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Arthur age is 30"},
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "00000000-0000-0000-0000-00000000000",
                        },
                    ],
                },
            ],
        ),
        (
            [
                AIMessage(
                    content=[
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "0",
                        }
                    ],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Rob"},
                            id="0",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Rob"},
                            "id": "0",
                        },
                    ],
                },
            ],
        ),
        (
            [
                AIMessage(
                    content=[
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "0",
                        }
                    ],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Rob"},
                            id="1",
                        ),
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Ben"},
                            "id": "0",
                        },
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "input": {"name": "Rob"},
                            "id": "1",
                        },
                    ],
                },
            ],
        ),
        (
            [
                AIMessage(
                    content=[
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "cache_control": {
                                "type": "ephemeral",
                            },
                            "input": {"name": "Ben"},
                            "id": "0",
                        },
                    ],
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Information",
                            "cache_control": {
                                "type": "ephemeral",
                            },
                            "input": {"name": "Ben"},
                            "id": "0",
                        },
                    ],
                },
            ],
        ),
        (
            [
                ToolMessage(
                    content="test",
                    tool_call_id="0",
                ),
            ],
            None,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "0",
                            "content": "test",
                        },
                    ],
                },
            ],
        ),
        (
            [
                AIMessage(
                    content_blocks=[
                        create_text_block(text="Text content"),
                        create_image_block(url="https://example.com/image.png"),
                        create_image_block(base64="/9j/4AAQSk", mime_type="image/png"),
                        create_image_block(file_id="1"),
                    ]
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Text content"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "/9j/4AAQSk",
                            },
                        },
                        {"type": "image", "source": {"type": "file", "file_id": "1"}},
                    ],
                }
            ],
        ),
        (
            [
                AIMessage(
                    content=[
                        {"type": "text", "text": "Text content"},
                        {
                            "type": "image",
                            "source_type": "url",
                            "url": "https://example.com/image.png",
                        },
                        {
                            "type": "image",
                            "source_type": "url",
                            "url": "data:image/png;base64,/9j/4AAQSk",
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "mime_type": "image/png",
                            "data": "/9j/4AAQSk",
                        },
                        {"type": "image", "source_type": "id", "id": "1"},
                    ]
                ),
            ],
            None,
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Text content"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "/9j/4AAQSk",
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "/9j/4AAQSk",
                            },
                        },
                        {"type": "image", "source": {"type": "file", "file_id": "1"}},
                    ],
                }
            ],
        ),
    ],
)
def test_format_messages_anthropic(
    source_history, expected_sm, expected_history
) -> None:
    """Test the original format_messages_anthropic functionality."""
    sm, result_history = _format_messages_anthropic(
        source_history, project="test-project"
    )

    for result, expected in zip(result_history, expected_history, strict=False):
        assert result == expected
    assert sm == expected_sm


def test_make_thinking_message_chunk_from_anthropic_event() -> None:
    """Test the conversion of Anthropic event into AIMessageChunk."""
    thinking_chunk = _make_message_chunk_from_anthropic_event(
        event=RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=ThinkingDelta(
                thinking="thoughts of the model...",
                type="thinking_delta",
            ),
        ),
        stream_usage=True,
        coerce_content_to_string=False,
    )
    signature_chunk = _make_message_chunk_from_anthropic_event(
        event=RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=SignatureDelta(
                signature="thoughts-signature",
                type="signature_delta",
            ),
        ),
        stream_usage=True,
        coerce_content_to_string=False,
    )

    assert thinking_chunk == AIMessageChunk(
        content=[
            {
                "index": 1,
                "type": "thinking",
                "thinking": "thoughts of the model...",
            }
        ]
    )
    assert signature_chunk == AIMessageChunk(
        content=[
            {
                "index": 1,
                "type": "thinking",
                "signature": "thoughts-signature",
            }
        ]
    )
    assert isinstance(thinking_chunk, AIMessageChunk)
    assert isinstance(signature_chunk, AIMessageChunk)


def test_thinking_in_params_true() -> None:
    """Test _thinking_in_params when thinking.type is 'enabled'."""
    params = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

    assert _thinking_in_params(params)


def test_thinking_in_params_false_different_type() -> None:
    """Test _thinking_in_params when thinking.type is 'disabled'."""
    params = {"thinking": {"type": "disabled", "budget_tokens": 1024}}

    assert not _thinking_in_params(params)


def test_documents_in_params_true() -> None:
    """Test _documents_in_params when document with citations is enabled."""
    params = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "document", "citations": {"enabled": True}}],
            }
        ]
    }

    assert _documents_in_params(params)


def test_documents_in_params_false_citations_disabled() -> None:
    """Test _documents_in_params when citations are not enabled."""
    params = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "document", "citations": {"enabled": False}}],
            }
        ]
    }

    assert not _documents_in_params(params)


def test_documents_in_params_false_no_document() -> None:
    """Test _documents_in_params when there are no documents."""
    params = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    }

    assert not _documents_in_params(params)


def test_ai_message_empty_content_with_tool_calls() -> None:
    """Test that AIMessage with empty content and tool_calls includes tool_calls output.

    Addresses the issue where tool_calls were being trimmed out when content was empty.
    """
    # Empty string content
    message_empty_string = AIMessage(
        content="",
        tool_calls=[
            create_tool_call(
                name="Information",
                args={"name": "Ben"},
                id="00000000-0000-0000-0000-00000000000",
            ),
        ],
    )

    result_empty_string = _format_message_anthropic(
        message_empty_string, project="test-project"
    )

    assert result_empty_string is not None
    assert result_empty_string["role"] == "assistant"
    assert "content" in result_empty_string
    content = result_empty_string["content"]

    tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1

    tool_use = tool_use_blocks[0]
    assert tool_use["name"] == "Information"
    assert tool_use["input"] == {"name": "Ben"}
    assert tool_use["id"] == "00000000-0000-0000-0000-00000000000"

    # Empty list content with tool_calls
    message_empty_list = AIMessage(
        content=[],
        tool_calls=[
            create_tool_call(
                name="GetWeather",
                args={"location": "New York"},
                id="11111111-1111-1111-1111-11111111111",
            ),
        ],
    )

    result_empty_list = _format_message_anthropic(
        message_empty_list, project="test-project"
    )

    assert result_empty_list is not None
    assert result_empty_list["role"] == "assistant"
    assert "content" in result_empty_list
    content = result_empty_list["content"]

    tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1

    tool_use = tool_use_blocks[0]
    assert tool_use["name"] == "GetWeather"
    assert tool_use["input"] == {"location": "New York"}
    assert tool_use["id"] == "11111111-1111-1111-1111-11111111111"

    # Whitespace-only content
    message = AIMessage(
        content="   \n\t  ",
        tool_calls=[
            create_tool_call(
                name="Calculator",
                args={"expression": "2 + 2"},
                id="22222222-2222-2222-2222-22222222222",
            ),
        ],
    )

    result = _format_message_anthropic(message, project="test-project")

    assert result is not None
    assert result["role"] == "assistant"
    assert "content" in result
    content = result["content"]

    # Should contain exactly one tool_use block (whitespace content is stripped)
    tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1

    tool_use = tool_use_blocks[0]
    assert tool_use["name"] == "Calculator"
    assert tool_use["input"] == {"expression": "2 + 2"}
    assert tool_use["id"] == "22222222-2222-2222-2222-22222222222"

    # Should not contain any text blocks (whitespace is stripped)
    text_blocks = [block for block in content if block.get("type") == "text"]
    assert len(text_blocks) == 0


def test_ai_message_empty_content_without_tool_calls() -> None:
    """Test AIMessage with empty content and no tool_calls properly returns None."""
    # Empty string content without tool_calls
    message_empty_string = AIMessage(content="")
    result_empty_string = _format_message_anthropic(
        message_empty_string, project="test-project"
    )
    assert result_empty_string is None

    # Empty list content without tool_calls
    message_empty_list = AIMessage(content=[])
    result_empty_list = _format_message_anthropic(
        message_empty_list, project="test-project"
    )
    assert result_empty_list is None


def test_format_messages_tool_message_with_streaming_metadata() -> None:
    """Test that streaming metadata is removed from ToolMessage content.

    Streaming adds 'index' and 'partial_json' fields that must be cleaned.
    """
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                create_tool_call(
                    name="get_weather", args={"city": "Paris"}, id="call_1"
                )
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "Sunny, 22°C",
                    "index": 0,
                }  # Streaming metadata
            ],
            tool_call_id="call_1",
        ),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # Verify tool_result format with no 'index' field
    assert formatted[1]["content"][0] == {
        "type": "tool_result",
        "content": [{"type": "text", "text": "Sunny, 22°C"}],  # NO 'index'
        "tool_use_id": "call_1",
    }


def test_format_messages_tool_message_with_error() -> None:
    """Test that error ToolMessages include is_error flag."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                create_tool_call(
                    name="get_weather", args={"city": "Paris"}, id="call_1"
                )
            ],
        ),
        ToolMessage(content="API key invalid", tool_call_id="call_1", status="error"),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # Verify is_error flag present
    assert formatted[1]["content"][0] == {
        "type": "tool_result",
        "content": "API key invalid",
        "tool_use_id": "call_1",
        "is_error": True,
    }


def test_format_messages_ai_message_with_streaming_metadata() -> None:
    """Test that AIMessage content blocks are cleaned of streaming metadata."""
    from langchain_core.messages import BaseMessage

    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {"type": "text", "text": "Calling tool...", "index": 0},
                {
                    "type": "tool_use",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                    "id": "call_1",
                    "index": 1,
                },
            ],
            tool_calls=[
                create_tool_call(
                    name="get_weather", args={"city": "Paris"}, id="call_1"
                )
            ],
        )
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # Verify 'index' removed from both blocks
    assert "index" not in formatted[0]["content"][0]
    # Tool use block should have 'index' removed but keep other fields
    tool_use_block = formatted[0]["content"][1]
    assert "index" not in tool_use_block
    assert tool_use_block["type"] == "tool_use"
    assert tool_use_block["name"] == "get_weather"


def test_format_messages_tool_message_with_partial_json() -> None:
    """Test that partial_json streaming metadata is removed."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                create_tool_call(name="calculator", args={"expr": "2+2"}, id="call_1")
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "4",
                    "partial_json": '{"result"',  # Streaming metadata
                }
            ],
            tool_call_id="call_1",
        ),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # Verify partial_json removed
    assert "partial_json" not in formatted[1]["content"][0]["content"][0]
    assert formatted[1]["content"][0]["content"][0] == {
        "type": "text",
        "text": "4",
    }


def test_format_messages_tool_message_backward_compatibility() -> None:
    """Test that already-formatted tool_result messages work (backward compat)."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                create_tool_call(
                    name="get_weather", args={"city": "Paris"}, id="call_1"
                )
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "tool_result",
                    "content": "Sunny, 22°C",
                    "tool_use_id": "call_1",
                }
            ],
            tool_call_id="call_1",
        ),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # Should pass through already-formatted tool_result
    assert formatted[1]["content"][0] == {
        "type": "tool_result",
        "content": "Sunny, 22°C",
        "tool_use_id": "call_1",
    }


def test_format_messages_complex_multiturn_with_tools() -> None:
    """Test complex multi-turn conversation with streaming metadata cleanup."""
    messages = [
        HumanMessage(content="What's the weather in Paris?"),
        AIMessage(
            content=[
                {"type": "text", "text": "I'll check that for you.", "index": 0},
            ],
            tool_calls=[
                create_tool_call(
                    name="get_weather", args={"city": "Paris"}, id="call_1"
                )
            ],
        ),
        ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "Sunny, 22°C",
                    "index": 0,
                    "partial_json": "{}",
                }
            ],
            tool_call_id="call_1",
        ),
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "It's sunny and 22°C in Paris!",
                    "index": 0,
                }
            ]
        ),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")

    # First message (human) should be unchanged
    assert formatted[0]["role"] == "user"

    # Second message (AI with tool call) should have 'index' removed
    assert "index" not in formatted[1]["content"][0]
    assert formatted[1]["content"][0]["text"] == "I'll check that for you."

    # Third message (tool result) should have streaming metadata removed
    tool_result = formatted[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert "index" not in tool_result["content"][0]
    assert "partial_json" not in tool_result["content"][0]

    # Fourth message (AI response) should have 'index' removed
    assert "index" not in formatted[3]["content"][0]
    assert formatted[3]["content"][0]["text"] == "It's sunny and 22°C in Paris!"


def test_tool_message_preserves_cache_control() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                create_tool_call(
                    name="get_weather",
                    args={"city": "Paris"},
                    id="call_1",
                )
            ],
        ),
        ToolMessage(
            content="Sunny, 22°C",
            tool_call_id="call_1",
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        ),
    ]

    _, formatted = _format_messages_anthropic(messages, project="test-project")
    tool_result = formatted[1]["content"][0]

    assert tool_result == {
        "type": "tool_result",
        "content": "Sunny, 22°C",
        "tool_use_id": "call_1",
        "cache_control": {"type": "ephemeral"},
    }


@pytest.mark.parametrize(
    ("image_url", "expected_media_type"),
    [
        ("https://example.com/image.png?token=123", "image/png"),
        ("https://example.com/image.jpg", "image/jpeg"),
        ("https://example.com/document.pdf", "application/pdf"),
    ],
)
def test_format_image(image_url: str, expected_media_type: str) -> None:
    """Test that _format_image correctly handles various URLs."""
    project = "test-project"

    with patch(
        "langchain_google_vertexai._anthropic_utils.ImageBytesLoader"
    ) as MockLoader:
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load_bytes.return_value = b"fake_image_data"

        result = _format_image(image_url, project)

        expected_data = base64.b64encode(b"fake_image_data").decode("ascii")

        assert result == {
            "type": "base64",
            "media_type": expected_media_type,
            "data": expected_data,
        }

        mock_loader_instance.load_bytes.assert_called_once_with(image_url)
