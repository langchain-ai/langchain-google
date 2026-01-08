"""**Callback handlers** allow listening to events in LangChain."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_google_community.callbacks.bigquery_callback import (
        AsyncBigQueryCallbackHandler,
        BigQueryCallbackHandler,
    )


_module_lookup = {
    "AsyncBigQueryCallbackHandler": (
        "langchain_google_community.callbacks.bigquery_callback"
    ),
    "BigQueryCallbackHandler": "langchain_google_community.callbacks.bigquery_callback",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "AsyncBigQueryCallbackHandler",
    "BigQueryCallbackHandler",
]
