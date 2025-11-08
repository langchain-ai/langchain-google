import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceAsyncClient as v1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceClient as v1PredictionServiceClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceAsyncClient as v1beta1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceClient as v1beta1PredictionServiceClient,
)

from langchain_google_vertexai._utils import get_client_info


@lru_cache
def _get_client_options(
    api_endpoint: str | None,
    cert_source: Callable[[], tuple[bytes, bytes]] | None,
) -> ClientOptions:
    """Return a shared `ClientOptions` object for each unique configuration."""
    client_options = ClientOptions(api_endpoint=api_endpoint)
    if cert_source:
        client_options.client_cert_source = cert_source
    return client_options


# Cache sync client
@lru_cache
def _get_prediction_client(
    *,
    endpoint_version: Literal["v1", "v1beta1"],
    credentials: Any,
    client_options: ClientOptions,
    transport: str | None,
    user_agent: str,
) -> v1PredictionServiceClient | v1beta1PredictionServiceClient:
    """Return a shared `PredictionServiceClient`."""
    client_kwargs: dict[str, Any] = {
        "credentials": credentials,
        "client_options": client_options,
        "client_info": get_client_info(module=user_agent),
        "transport": transport,
    }
    if endpoint_version == "v1":
        return v1PredictionServiceClient(**client_kwargs)
    return v1beta1PredictionServiceClient(**client_kwargs)


# Cache async client - must store caches per event loop
_client_caches: WeakKeyDictionary = WeakKeyDictionary()


def _create_async_prediction_client(
    *,
    endpoint_version: Literal["v1", "v1beta1"],
    credentials: Any,
    client_options: ClientOptions,
    transport: str | None,
    user_agent: str,
) -> v1PredictionServiceAsyncClient | v1beta1PredictionServiceAsyncClient:
    """Create a new `PredictionServiceAsyncClient`."""
    # async clients don't support "rest" transport with standard Google APIs
    # https://github.com/googleapis/gapic-generator-python/issues/1962
    # However, when using custom endpoints, we can try to keep REST transport
    has_custom_endpoint = False
    if client_options.api_endpoint:
        try:
            endpoint = client_options.api_endpoint
            # Add scheme if missing for proper URL parsing
            if not endpoint.startswith(("http://", "https://")):
                endpoint = f"https://{endpoint}"

            parsed_url = urlparse(endpoint)
            hostname = parsed_url.hostname or ""
            # Check if hostname matches aiplatform.googleapis.com (exact or regional)
            has_custom_endpoint = not (
                hostname == "aiplatform.googleapis.com"
                or hostname.endswith("-aiplatform.googleapis.com")
            )
        except Exception:
            # If URL parsing fails, treat as custom endpoint for safety
            has_custom_endpoint = True

    # Use grpc_asyncio for better async performance, except with custom endpoints
    if not has_custom_endpoint and transport in (None, "grpc", "rest"):
        transport = "grpc_asyncio"

    async_client_kwargs: dict[str, Any] = {
        "client_options": client_options,
        "client_info": get_client_info(module=user_agent),
        "credentials": credentials,
        "transport": transport,
    }

    if endpoint_version == "v1":
        return v1PredictionServiceAsyncClient(**async_client_kwargs)
    return v1beta1PredictionServiceAsyncClient(**async_client_kwargs)


def _get_async_prediction_client(
    *,
    endpoint_version: Literal["v1", "v1beta1"],
    credentials: Any,
    client_options: ClientOptions,
    transport: str | None,
    user_agent: str,
):
    """Return a shared PredictionServiceAsyncClient per event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no event loop is running, don't cache
        return _create_async_prediction_client(
            endpoint_version=endpoint_version,
            credentials=credentials,
            client_options=client_options,
            transport=transport,
            user_agent=user_agent,
        )

    # Get or create cache for this event loop
    if loop not in _client_caches:
        _client_caches[loop] = {}

    cache_key = (
        endpoint_version,
        id(credentials),
        id(client_options),
        transport,
        user_agent,
    )

    if cache_key not in _client_caches[loop]:
        _client_caches[loop][cache_key] = _create_async_prediction_client(
            endpoint_version=endpoint_version,
            credentials=credentials,
            client_options=client_options,
            transport=transport,
            user_agent=user_agent,
        )

    return _client_caches[loop][cache_key]
