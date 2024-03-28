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

import re


def extract_model_response(text, prompt):
    # Remove the "Prompt:\n<prompt>\n" section from the start
    prompt_section_pattern = re.compile(r'^Prompt:\n' + re.escape(prompt) + r'\n', re.DOTALL)
    text_without_prompt_section = prompt_section_pattern.sub('', text, count=1)
    
    # Define the output section starting pattern to look for
    output_start_pattern = re.compile(r'^Output:\n', re.DOTALL)

    # Check if the section immediately following "Output:\n" is the prompt
    if re.match(output_start_pattern.pattern + re.escape(prompt), text_without_prompt_section):
        # If the prompt is indeed repeated, remove "Output:\n<prompt>\n"
        output_without_repeated_prompt = re.sub(output_start_pattern.pattern + re.escape(prompt), '', text_without_prompt_section, count=1)
    else:
        # If the prompt is not repeated, simply remove "Output:\n" to start extracting the model response
        output_without_repeated_prompt = re.sub(output_start_pattern.pattern, '', text_without_prompt_section, count=1)

    # Return the cleaned output section, which is the model response
    return output_without_repeated_prompt.strip()


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
        
        if not kwargs.get("keep_original_response", False):
            response.predictions[0] = extract_model_response(response.predictions[0], prompts[0])
            
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
        
        if not kwargs.get("keep_original_response", False):
            response.predictions[0] = extract_model_response(response.predictions[0], prompts[0])
            
        return self._parse_response(response)
