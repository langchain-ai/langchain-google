import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class VertexAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks VertexAI info."""

    prompt_tokens: int = 0
    prompt_characters: int = 0
    completion_tokens: int = 0
    completion_characters: int = 0
    successful_requests: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"\tPrompt tokens: {self.prompt_tokens}\n"
            f"\tPrompt characters: {self.prompt_characters}\n"
            f"\tCompletion tokens: {self.completion_tokens}\n"
            f"\tCompletion characters: {self.completion_characters}\n"
            f"\tCached tokens: {self.cached_tokens}\n"
            f"\tTotal tokens: {self.total_tokens}\n"
            f"Successful requests: {self.successful_requests}\n"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Runs when LLM starts running."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Runs on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collects token usage."""
        completion_tokens, prompt_tokens, total_tokens, cached_tokens = 0, 0, 0, 0
        completion_characters, prompt_characters = 0, 0
        for generations in response.generations:
            if len(generations) > 0 and generations[0].generation_info:
                usage_metadata = generations[0].generation_info.get(
                    "usage_metadata", {}
                )
                completion_tokens += usage_metadata.get("candidates_token_count", 0)
                prompt_tokens += usage_metadata.get("prompt_token_count", 0)
                total_tokens += usage_metadata.get("total_token_count", 0)
                cached_tokens += usage_metadata.get("cached_content_token_count", 0)
                completion_characters += usage_metadata.get(
                    "candidates_billable_characters", 0
                )
                prompt_characters += usage_metadata.get("prompt_billable_characters", 0)

        with self._lock:
            self.prompt_characters += prompt_characters
            self.prompt_tokens += prompt_tokens
            self.completion_characters += completion_characters
            self.completion_tokens += completion_tokens
            self.successful_requests += 1
            self.total_tokens += total_tokens
            self.cached_tokens += cached_tokens
