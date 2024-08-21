from typing import Any, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_mistralai import (  # type: ignore[unused-ignore, import-not-found]
    chat_models,
)

from langchain_google_vertexai.model_garden_maas._base import (
    _BaseVertexMaasModelGarden,
    acompletion_with_retry,
    completion_with_retry,
)

chat_models.acompletion_with_retry = acompletion_with_retry  # type: ignore[unused-ignore, assignment]


class VertexModelGardenMistral(_BaseVertexMaasModelGarden, chat_models.ChatMistralAI):  # type: ignore[unused-ignore, misc]
    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        return completion_with_retry(self, **kwargs)
