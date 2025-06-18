import asyncio
from functools import lru_cache
from typing import Optional
from weakref import WeakKeyDictionary, WeakSet

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
_clients_to_close: WeakSet = WeakSet()


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
    client = genaix.build_generative_async_service(
        credentials=None,
        api_key=api_key,
        client_info=get_client_info(f"ChatGoogleGenerativeAI:{model}"),
        client_options=None,
        transport=transport,
    )
    _clients_to_close.add(client)
    return client


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


# import atexit

# import asyncio, atexit

# _cleanup_loop = asyncio.new_event_loop()

# async def _close_everything():
#     await asyncio.gather(
#         *(c.transport.close() for c in _clients_to_close),
#         return_exceptions=True,
#     )
#     # let grpc.aio finish any pending callbacks
#     await asyncio.sleep(0)

# def _shutdown_all_clients():
#     try:
#         _cleanup_loop.run_until_complete(_close_everything())
#         _cleanup_loop.run_until_complete(_cleanup_loop.shutdown_asyncgens())
#     finally:
#         _cleanup_loop.close()

# atexit.register(_shutdown_all_clients)
