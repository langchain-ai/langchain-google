import asyncio
from functools import lru_cache
from typing import Any, Literal, Optional, Union
from weakref import WeakKeyDictionary

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
