"""Integration tests for Anthropic cache control functionality."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_vertexai.model_garden import ChatAnthropicVertex


@pytest.mark.extended
@pytest.mark.skip(reason="claude-3-5-v2 not enabled")
def test_anthropic_system_cache() -> None:
    """Test chat with system message having cache control."""
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    context = SystemMessage(
        content="You are my personal assistant. Be helpful and concise.",
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )
    message = HumanMessage(content="Hello! What can you do for me?")

    response = model.invoke([context, message], model_name="claude-sonnet-4-5@20250929")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "usage_metadata" in response.additional_kwargs
    assert "cache_creation_input_tokens" in response.additional_kwargs["usage_metadata"]


@pytest.mark.extended
@pytest.mark.skip(reason="claude-3-5-v2 not enabled")
def test_anthropic_mixed_cache() -> None:
    """Test chat with different cache control types."""
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    context = SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are my personal assistant.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "What's your name and what can you help me with?",
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )

    response = model.invoke([context, message], model_name="claude-sonnet-4-5@20250929")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "usage_metadata" in response.additional_kwargs


@pytest.mark.extended
@pytest.mark.skip(reason="claude-3-5-v2 not enabled")
def test_anthropic_conversation_cache() -> None:
    """Test chat conversation with cache control."""
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    context = SystemMessage(
        content="You are my personal assistant. My name is Peter.",
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
    )
    messages = [
        context,
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What's my name?",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
        AIMessage(content="Your name is Peter."),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Can you repeat my name?",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
    ]

    response = model.invoke(messages, model_name="claude-sonnet-4-5@20250929")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "peter" in response.content.lower()  # Should remember the name


@pytest.mark.extended
@pytest.mark.skip(reason="claude-3-5-v2 not enabled")
def test_anthropic_chat_template_cache() -> None:
    """Test chat template with structured content and cache control."""
    project = os.environ["PROJECT_ID"]
    location = "us-central1"
    model = ChatAnthropicVertex(
        project=project,
        location=location,
    )

    content: list[dict[str, str | dict[str, str]] | str] = [
        {
            "text": "You are a helpful assistant. Be concise and clear.",
            "type": "text",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=content), ("human", "{input}")]
    )

    chain = prompt | model

    response = chain.invoke(
        {"input": "What's the capital of France?"},
    )

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content
