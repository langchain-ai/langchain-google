import asyncio
from functools import lru_cache
from typing import Optional
from weakref import WeakKeyDictionary

from google.ai.generativelanguage_v1beta import (
    GenerativeServiceAsyncClient as v1betaGenerativeServiceAsyncClient,
)
from google.ai.generativelanguage_v1beta import (
    GenerativeServiceClient as v1betaGenerativeServiceClient,
)

from langchain_google_genai._common import get_client_info

from . import _genai_extension as genaix


# Cache sync client
@lru_cache
def _get_sync_client(
    *,
    api_key: Optional[str],
    model: str,
    transport: str,
) -> v1betaGenerativeServiceClient:
    """Return a shared sync client."""
    client_info = get_client_info(f"ChatGoogleGenerativeAI:{model}")
    return genaix.build_generative_service(
        api_key=api_key,
        client_info=client_info,
        client_options=None,
        transport=transport,
    )


# Cache async client - must store caches per event loop
_client_caches: WeakKeyDictionary = WeakKeyDictionary()


def _create_async_client(
    *,
    api_key: Optional[str],
    model: str,
    transport: Optional[str],
) -> v1betaGenerativeServiceAsyncClient:
    """Create a new async client."""
    # async clients don't support "rest" transport
    # https://github.com/googleapis/gapic-generator-python/issues/1962
    if transport == "rest":
        transport = "grpc_asyncio"
    return genaix.build_generative_async_service(
        credentials=None,
        api_key=api_key,
        client_info=get_client_info(f"ChatGoogleGenerativeAI:{model}"),
        client_options=None,
        transport=transport,
    )


def _get_async_client(
    *,
    api_key: Optional[str],
    model: str,
    transport: Optional[str],
) -> v1betaGenerativeServiceAsyncClient:
    """Return a shared async client per event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no event loop is running, don't cache
        return _create_async_client(
            api_key=api_key,
            model=model,
            transport=transport,
        )

    # Get or create cache for this event loop
    if loop not in _client_caches:
        _client_caches[loop] = {}

    cache_key = (api_key, model, transport)

    if cache_key not in _client_caches[loop]:
        _client_caches[loop][cache_key] = _create_async_client(
            api_key=api_key,
            model=model,
            transport=transport,
        )

    return _client_caches[loop][cache_key]
