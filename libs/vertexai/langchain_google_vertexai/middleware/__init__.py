"""Middleware for Vertex AI integrations."""

from langchain_google_vertexai.middleware.prompt_caching import (
    VertexPromptCachingMiddleware,
)

__all__ = ["VertexPromptCachingMiddleware"]
