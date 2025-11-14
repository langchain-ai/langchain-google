from functools import lru_cache
from typing import Any, Optional

from google.api_core.client_options import ClientOptions
from google.auth import credentials
from google.cloud.modelarmor_v1 import ModelArmorClient

_DEFAULT_LOCATION = "us-central1"


@lru_cache
def _get_model_armor_client(
    location: str = _DEFAULT_LOCATION,
    credentials: Optional[credentials.Credentials] = None,
    transport: Optional[str] = None,
    client_options: Optional[ClientOptions] = None,
    client_info: Optional[Any] = None,
) -> ModelArmorClient:
    """
    Initialize the Model Armor client.

    Args:
        location: The location of the Model Armor client.
        credentials: Credentials to use when making API calls.
        transport: Desired API transport method, can be either `'grpc'` or `'rest'`.
        client_options: Client options for the API client.
        client_info: Client info for the API client.

    Returns:
        ModelArmorClient: The Model Armor client.
    """
    if client_options is None:
        client_options = ClientOptions(
            api_endpoint=f"modelarmor.{location}.rep.googleapis.com"
        )

    # Build the client constructor arguments.
    client_kwargs: dict = {}

    if client_options is not None:
        client_kwargs["client_options"] = client_options

    if credentials is not None:
        client_kwargs["credentials"] = credentials

    if transport is not None:
        client_kwargs["transport"] = transport

    if client_info is not None:
        client_kwargs["client_info"] = client_info

    return ModelArmorClient(**client_kwargs)
