from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult

from langchain_google_vertexai._base import _BaseVertexAIModelGarden


class VertexAIModelGarden(_BaseVertexAIModelGarden, BaseLLM):
    """Large language models served from Vertex AI Model Garden."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)

        if self.single_example_per_request and len(instances) > 1:
            results = []
            for instance in instances:
                response = self.client.predict(
                    endpoint=self.endpoint_path, instances=[instance]
                )
                results.append(self._parse_prediction(response.predictions[0]))
            return LLMResult(
                generations=[[Generation(text=result)] for result in results]
            )

        response = self.client.predict(endpoint=self.endpoint_path, instances=instances)
        return self._parse_response(response)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        instances = self._prepare_request(prompts, **kwargs)
        if self.single_example_per_request and len(instances) > 1:
            responses = []
            for instance in instances:
                responses.append(
                    self.async_client.predict(
                        endpoint=self.endpoint_path, instances=[instance]
                    )
                )

            responses = await asyncio.gather(*responses)
            return LLMResult(
                generations=[
                    [Generation(text=self._parse_prediction(response.predictions[0]))]
                    for response in responses
                ]
            )

        response = await self.async_client.predict(
            endpoint=self.endpoint_path, instances=instances
        )
        return self._parse_response(response)
