"""Unit tests for `langchain_google_genai.utils`."""

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langchain_google_genai.utils import create_context_cache


def test_create_context_cache_applies_tool_choice() -> None:
    """`tool_choice` must be wired into the cache's `tool_config` (#1773 area).

    Previously the documented `tool_choice` parameter was accepted and silently
    dropped, so a forced/among tool selection never reached the cached content.
    """

    @tool
    def search(q: str) -> str:
        """Search for something."""
        return q

    model = MagicMock()
    model.model = "gemini-2.5-flash"
    model._extract_tool_names.return_value = ["search"]

    captured: dict[str, Any] = {}

    def fake_create(*, model: Any, config: Any) -> Any:
        captured["config"] = config
        result = MagicMock()
        result.name = "caches/abc"
        return result

    model.client.caches.create.side_effect = fake_create

    name = create_context_cache(
        model,
        [HumanMessage("hi")],
        tools=[search],
        tool_choice="any",
    )

    assert name == "caches/abc"
    # tool_choice must be applied to the cache's tool_config, not dropped.
    assert captured["config"].tool_config is not None
    assert captured["config"].tool_config.function_calling_config is not None
