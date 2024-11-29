import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call

from langchain_google_vertexai.model_garden import _format_messages_anthropic


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
            "test1",
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
                            "type": "image_url",
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
    sm, result_history = _format_messages_anthropic(source_history)

    for result, expected in zip(result_history, expected_history):
        assert result == expected
    assert sm == expected_sm
