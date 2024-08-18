from typing import Any, Optional

import httpx
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_mistralai import chat_models

from langchain_google_vertexai.model_garden_maas._base import (
    _BaseVertexMaasModelGarden,
    _get_token,
    acompletion_with_retry,
    completion_with_retry,
)

chat_models.acompletion_with_retry = acompletion_with_retry  # type: ignore[assignment]


class VertexModelGardenMistral(_BaseVertexMaasModelGarden, chat_models.ChatMistralAI):  # type: ignore[misc]
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        token = _get_token(credentials=self.credentials)
        self.endpoint = self.get_url()
        self.client = httpx.Client(
            base_url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "x-goog-api-client": self._library_version,
                "user_agent": self._user_agent,
            },
            timeout=self.timeout,
        )
        self.async_client = httpx.AsyncClient(
            base_url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "x-goog-api-client": self._library_version,
                "user_agent": self._user_agent,
            },
            timeout=self.timeout,
        )

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        return completion_with_retry(self, **kwargs)
