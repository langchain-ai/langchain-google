from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM, LangSmithParams
from langchain_core.messages import HumanMessage
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_google_vertexai._base import GoogleModelFamily, _VertexAICommon
from langchain_google_vertexai.chat_models import ChatVertexAI


class VertexAI(_VertexAICommon, BaseLLM):
    """Google Vertex AI large language models."""

    model_name: str = Field(default="gemini-2.0-flash-001", alias="model")
    "The name of the Vertex AI large language model."
    tuned_model_name: Optional[str] = None
    """The name of a tuned model. If tuned_model_name is passed
    model_name will be used to determine the model family
    """

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg."""
        if model_name:
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "vertexai"]

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the python package exists in environment."""
        if "medlm" in self.model_name and self.model_family == GoogleModelFamily.PALM:
            err = (
                "MedLM on Palm is not supported any more! Please, use Gemini or "
                "switch to langchain-google-vertexai==2.0.13"
            )
            raise ValueError(err)
        self.client = ChatVertexAI(
            model_name=self.model_name,
            tuned_model_name=self.tuned_model_name,
            project=self.project,
            location=self.location,
            credentials=self.credentials,
            api_transport=self.api_transport,
            api_endpoint=self.api_endpoint,
            default_metadata=self.default_metadata,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            safety_settings=self.safety_settings,
            n=self.n,
            seed=self.seed,
        )
        return self

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._prepare_params(stop=stop, **kwargs)
        ls_params = super()._get_ls_params(stop=stop, **params)
        ls_params["ls_provider"] = "google_vertexai"
        if ls_max_tokens := params.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        for prompt in prompts:
            chat_result = self.client._generate(
                [HumanMessage(content=prompt)],
                stop=stop,
                stream=stream,
                run_manager=run_manager,
                **kwargs,
            )

            generations.append(
                [
                    Generation(
                        text=g.message.content,
                        generation_info={**g.generation_info},
                    )
                    for g in chat_result.generations
                ]
            )
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []
        for prompt in prompts:
            chat_result = await self.client._agenerate(
                [HumanMessage(content=prompt)],
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            generations.append(
                [
                    Generation(
                        text=g.message.content,
                        generation_info={
                            **g.generation_info,
                        },
                    )
                    for g in chat_result.generations
                ]
            )
        return LLMResult(generations=generations)

    @staticmethod
    def _lc_usage_to_metadata(lc_usage: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "input_tokens": "prompt_token_count",
            "output_tokens": "candidates_token_count",
            "total_tokens": "total_token_count",
        }
        return {mapping[k]: v for k, v in lc_usage.items() if v and k in mapping}

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_chunk in self.client._stream(
            [HumanMessage(content=prompt)],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            if stream_chunk.message.usage_metadata:
                lc_usage = stream_chunk.message.usage_metadata
                usage_metadata = {
                    **lc_usage,
                    **self._lc_usage_to_metadata(lc_usage=lc_usage),
                }
            else:
                usage_metadata = {}
            chunk = GenerationChunk(
                text=stream_chunk.message.content,
                generation_info={
                    **stream_chunk.generation_info,
                    **{"usage_metadata": usage_metadata},
                },
            )
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for stream_chunk in self.client._astream(
            [HumanMessage(content=prompt)],
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            chunk = GenerationChunk(text=stream_chunk.message.content)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text, chunk=chunk, verbose=self.verbose
                )

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        return self.client.get_num_tokens(text)
