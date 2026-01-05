"""
Model Armor Middleware for LangChain Agents.

This module provides middleware for integrating Model Armor sanitization
into LangChain agents created with the create_agent API.
"""

import logging
from typing import Any, Optional

import langchain.agents as lc_agents
import langchain.agents.middleware as lc_agents_middleware
import langchain.agents.middleware.types as lc_hook_types
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

logger = logging.getLogger(__name__)


class ModelArmorMiddleware(lc_agents_middleware.AgentMiddleware):
    """
    Middleware to integrate Model Armor sanitization into agent execution.

    This middleware provides hooks that sanitize user prompts before they reach
    the model and sanitize model responses before they're returned to the user.

    Sanitization is enabled by providing the corresponding runnable:
    - Provide `prompt_sanitizer` to enable user prompt sanitization
    - Provide `response_sanitizer` to enable model response sanitization

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_google_vertexai import ChatVertexAI
        from langchain_google_community.model_armor import (
            ModelArmorMiddleware,
            ModelArmorSanitizePromptRunnable,
            ModelArmorSanitizeResponseRunnable,
        )

        # Create sanitizer runnables
        prompt_sanitizer = ModelArmorSanitizePromptRunnable(
            project="my-project",
            location="us-central1",
            template_id="my-template",
            fail_open=False,
        )
        response_sanitizer = ModelArmorSanitizeResponseRunnable(
            project="my-project",
            location="us-central1",
            template_id="my-template",
            fail_open=False,
        )

        # Create middleware with both sanitizers
        middleware = ModelArmorMiddleware(
            prompt_sanitizer=prompt_sanitizer,
            response_sanitizer=response_sanitizer,
        )

        # Create agent with Model Armor protection
        agent = create_agent(
            model=ChatVertexAI(model_name="gemini-2.0-flash-001"),
            tools=[...],
            middleware=[middleware],
        )

        # Or create middleware with only prompt sanitization
        middleware = ModelArmorMiddleware(prompt_sanitizer=prompt_sanitizer)
        ```

    Args:
        prompt_sanitizer: Runnable for sanitizing user prompts before model calls.
            If provided, prompt sanitization is enabled. If None, prompts are
            not sanitized.
        response_sanitizer: Runnable for sanitizing model responses.
            If provided, response sanitization is enabled. If None, responses
            are not sanitized.
    """

    def __init__(
        self,
        *,
        prompt_sanitizer: ModelArmorSanitizePromptRunnable | None = None,
        response_sanitizer: ModelArmorSanitizeResponseRunnable | None = None,
    ):
        """Initialize Model Armor middleware.

        Args:
            prompt_sanitizer: Runnable for sanitizing user prompts. If provided,
                prompt sanitization is enabled.
            response_sanitizer: Runnable for sanitizing model responses. If
                provided, response sanitization is enabled.

        Raises:
            ValueError: If neither prompt_sanitizer nor response_sanitizer is
                provided (middleware would have no effect).
        """
        if prompt_sanitizer is None and response_sanitizer is None:
            msg = (
                "At least one of prompt_sanitizer or response_sanitizer must be "
                "provided. Without sanitizers, the middleware has no effect."
            )
            raise ValueError(msg)

        self.prompt_sanitizer = prompt_sanitizer
        self.response_sanitizer = response_sanitizer

    @lc_hook_types.hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: lc_agents.AgentState,
        runtime: Runtime,
    ) -> Optional[dict[str, Any]]:
        """
        Sanitize user prompts before sending to the model.

        This hook is called before the model processes the input.
        It extracts the latest user message and sanitizes it using Model Armor.

        Args:
            state: Current agent state containing messages.
            runtime: Runtime object containing context and configuration.

        Returns:
            None if content is safe, or dict with jump_to="end" if unsafe.

        Raises:
            ValueError: If content is unsafe and fail_open=False. This is caught
                internally and converted to a jump_to="end" response.
        """
        if self.prompt_sanitizer is None:
            return None

        messages = state["messages"]
        if not messages:
            return None

        # Get the last message - this is the most recent user input before model call
        last_message = messages[-1]

        # Sanitize the message content
        try:
            # The sanitizer will raise ValueError if content is unsafe
            # and fail_open=False
            self.prompt_sanitizer.invoke(last_message)
            content_preview = str(last_message.content)[:50]
            logger.debug("Prompt sanitization passed for: %s...", content_preview)
            return None
        except ValueError as e:
            logger.debug("Prompt blocked by Model Armor Middleware: %s", e)

            # Jump to end instead of raising exception (per LangChain guardrails
            # pattern).
            return {
                "messages": [
                    AIMessage(
                        "I cannot process that request due to content policy "
                        "violations."
                    )
                ],
                "jump_to": "end",
            }

    @lc_hook_types.hook_config(can_jump_to=["end"])
    def after_model(
        self,
        state: lc_agents.AgentState,
        runtime: Runtime,
    ) -> Optional[dict[str, Any]]:
        """
        Sanitize model responses before returning to the user.

        This hook is called after the model generates a response.
        We sanitize the AI's response to ensure it doesn't contain
        harmful content.

        Args:
            state: Current agent state containing messages.
            runtime: Runtime object containing context and configuration.

        Returns:
            None if content is safe, or dict with jump_to="end" if unsafe.

        Raises:
            ValueError: If content is unsafe and fail_open=False. This is caught
                internally and converted to a jump_to="end" response.
        """
        if self.response_sanitizer is None:
            return None

        messages = state["messages"]
        if not messages:
            return None

        # Get the last message - this is the model's response after generation
        last_message = messages[-1]

        # Only sanitize messages with content (skip tool calls without content)
        if hasattr(last_message, "content") and last_message.content:
            try:
                self.response_sanitizer.invoke(last_message)
                logger.debug("Response sanitization passed")
                return None
            except ValueError as e:
                logger.debug("Response blocked by Model Armor Middleware: %s", e)

                # Jump to end instead of raising exception (per LangChain guardrails
                # pattern).
                return {
                    "messages": [
                        AIMessage(
                            "I cannot provide that response due to content "
                            "policy violations."
                        )
                    ],
                    "jump_to": "end",
                }

        return None


__all__ = ["ModelArmorMiddleware"]
