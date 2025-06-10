import asyncio
from functools import lru_cache
from typing import Any, Callable, Literal, Optional, Tuple, Union
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
    api_endpoint: Optional[str],
    cert_source: Optional[Callable[[], Tuple[bytes, bytes]]],
) -> ClientOptions:
    """Return a shared ClientOptions object for each unique configuration."""
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
    transport: str,
    user_agent: str,
) -> Union[v1PredictionServiceClient, v1beta1PredictionServiceClient]:
    """Return a shared PredictionServiceClient."""
    client_kwargs: dict[str, Any] = {
        "credentials": credentials,
        "client_options": client_options,
        "client_info": get_client_info(module=user_agent),
        "transport": transport,
    }
    if endpoint_version == "v1":
        return v1PredictionServiceClient(**client_kwargs)
    else:
        return v1beta1PredictionServiceClient(**client_kwargs)


# Cache async client - must store caches per event loop
_client_caches: WeakKeyDictionary = WeakKeyDictionary()


def _create_async_prediction_client(
    *,
    endpoint_version: Literal["v1", "v1beta1"],
    credentials: Any,
    client_options: ClientOptions,
    user_agent: str,
) -> Union[v1PredictionServiceAsyncClient, v1beta1PredictionServiceAsyncClient]:
    """Create a new PredictionServiceAsyncClient."""
    async_client_kwargs: dict[str, Any] = {
        "client_options": client_options,
        "client_info": get_client_info(module=user_agent),
        "credentials": credentials,
        # async clients don't support "rest" transport
        # https://github.com/googleapis/gapic-generator-python/issues/1962
        "transport": "grpc_asyncio",
    }

    if endpoint_version == "v1":
        return v1PredictionServiceAsyncClient(**async_client_kwargs)
    else:
        return v1beta1PredictionServiceAsyncClient(**async_client_kwargs)


def _get_async_prediction_client(
    *,
    endpoint_version: Literal["v1", "v1beta1"],
    credentials: Any,
    client_options: ClientOptions,
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
            user_agent=user_agent,
        )

    # Get or create cache for this event loop
    if loop not in _client_caches:
        _client_caches[loop] = {}

    cache_key = (endpoint_version, id(credentials), id(client_options), user_agent)

    if cache_key not in _client_caches[loop]:
        _client_caches[loop][cache_key] = _create_async_prediction_client(
            endpoint_version=endpoint_version,
            credentials=credentials,
            client_options=client_options,
            user_agent=user_agent,
        )

    return _client_caches[loop][cache_key]
