from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from vertexai.generative_models import ToolConfig

from langchain_google_vertexai.utils import create_context_cache


def test_create_context_cache_wraps_tool_config_for_sdk() -> None:
    """`CachedContent.create` requires the SDK `ToolConfig`, not the raw GAPIC
    type `_format_tool_config` returns, or it raises
    `TypeError: tool_config must be a ToolConfig object.` (issue #1241)."""
    model = MagicMock()
    model.project = "test-project"
    model.full_model_name = "publishers/google/models/gemini-2.0-flash"

    with patch(
        "langchain_google_vertexai.utils.caching.CachedContent.create"
    ) as mock_create:
        mock_create.return_value = MagicMock(name="cached_content")

        create_context_cache(
            model,
            messages=[HumanMessage(content="hello")],
            tool_config={
                "function_calling_config": {
                    "mode": 1,
                    "allowed_function_names": ["my_tool"],
                }
            },
        )

    passed_tool_config = mock_create.call_args.kwargs["tool_config"]
    assert isinstance(passed_tool_config, ToolConfig)
