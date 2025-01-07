"""Unit tests for _anthropic_utils.py."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call

from langchain_google_vertexai._anthropic_utils import (
    _format_message_anthropic,
    _format_messages_anthropic,
)


def test_format_message_anthropic_with_cache_control_in_kwargs():
    """Test formatting a message with cache control in additional_kwargs."""
    message = HumanMessage(
        content="Hello", additional_kwargs={"cache_control": {"type": "semantic"}}
    )
    result = _format_message_anthropic(message)
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ],
    }


def test_format_message_anthropic_with_cache_control_in_block():
    """Test formatting a message with cache control in content block."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ]
    )
    result = _format_message_anthropic(message)
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}}
        ],
    }


def test_format_message_anthropic_with_mixed_blocks():
    """Test formatting a message with mixed blocks, some with cache control."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}},
            {"type": "text", "text": "World"},
            "Plain text",
        ]
    )
    result = _format_message_anthropic(message)
    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "semantic"}},
            {"type": "text", "text": "World"},
            {"type": "text", "text": "Plain text"},
        ],
    }


def test_format_messages_anthropic_with_system_cache_control():
    """Test formatting messages with system message having cache control."""
    messages = [
        SystemMessage(
            content="System message",
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        ),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(messages)

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


def test_format_message_anthropic_system():
    """Test formatting a system message."""
    message = SystemMessage(
        content="System message",
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )
    result = _format_message_anthropic(message)
    assert result == [
        {
            "type": "text",
            "text": "System message",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_format_message_anthropic_system_list():
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
    result = _format_message_anthropic(message)
    assert result == [
        {
            "type": "text",
            "text": "System rule 1",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "System rule 2"},
    ]


def test_format_messages_anthropic_with_system_string():
    """Test formatting messages with system message as string."""
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Hello"),
    ]
    system_messages, formatted_messages = _format_messages_anthropic(messages)

    assert system_messages == [{"type": "text", "text": "System message"}]

    assert formatted_messages == [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]


def test_format_messages_anthropic_with_system_list():
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
    system_messages, formatted_messages = _format_messages_anthropic(messages)

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


def test_format_messages_anthropic_with_system_mixed_list():
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
    system_messages, formatted_messages = _format_messages_anthropic(messages)

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


def test_format_messages_anthropic_with_mixed_messages():
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
    system_messages, formatted_messages = _format_messages_anthropic(messages)

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
    "source_history, expected_sm, expected_history",
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
    ],
)
def test_format_messages_anthropic(
    source_history, expected_sm, expected_history
) -> None:
    """Test the original format_messages_anthropic functionality."""
    sm, result_history = _format_messages_anthropic(source_history)

    for result, expected in zip(result_history, expected_history):
        assert result == expected
    assert sm == expected_sm
