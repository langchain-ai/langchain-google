from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from google.cloud.aiplatform import telemetry
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import root_validator
from vertexai.generative_models import (  # type: ignore[import-untyped]
    Candidate,
    GenerativeModel,
    Image,
)
from vertexai.language_models import (  # type: ignore[import-untyped]
    CodeGenerationModel,
    TextGenerationModel,
)
from vertexai.language_models._language_models import (  # type: ignore[import-untyped]
    TextGenerationResponse,
)
from vertexai.preview.language_models import (  # type: ignore[import-untyped]
    CodeGenerationModel as PreviewCodeGenerationModel,
)
from vertexai.preview.language_models import (
    TextGenerationModel as PreviewTextGenerationModel,
)

from langchain_google_vertexai._base import (
    _VertexAICommon,
)
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
    is_codey_model,
    is_gemini_model,
)


def _completion_with_retry(
    llm: VertexAI,
    prompt: List[Union[str, Image]],
    stream: bool = False,
    is_gemini: bool = False,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    def _completion_with_retry_inner(
        prompt: List[Union[str, Image]], is_gemini: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return llm.client.generate_content(
                prompt,
                stream=stream,
                safety_settings=kwargs.pop("safety_settings", None),
                generation_config=kwargs,
            )
        else:
            if stream:
                return llm.client.predict_streaming(prompt[0], **kwargs)
            return llm.client.predict(prompt[0], **kwargs)

    with telemetry.tool_context_manager(llm._user_agent):
        return _completion_with_retry_inner(prompt, is_gemini, **kwargs)


async def _acompletion_with_retry(
    llm: VertexAI,
    prompt: str,
    is_gemini: bool = False,
    stream: bool = False,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = create_retry_decorator(
        max_retries=llm.max_retries, run_manager=run_manager
    )

    @retry_decorator
    async def _acompletion_with_retry_inner(
        prompt: str, is_gemini: bool = False, stream: bool = False, **kwargs: Any
    ) -> Any:
        if is_gemini:
            return await llm.client.generate_content_async(
                prompt,
                generation_config=kwargs,
                stream=stream,
                safety_settings=kwargs.pop("safety_settings", None),
            )
        if stream:
            raise ValueError("Async streaming is supported only for Gemini family!")
        return await llm.client.predict_async(prompt, **kwargs)

    with telemetry.tool_context_manager(llm._user_agent):
        return await _acompletion_with_retry_inner(
            prompt, is_gemini, stream=stream, **kwargs
        )


class VertexAI(_VertexAICommon, BaseLLM):
    """Google Vertex AI large language models."""

    model_name: str = "text-bison"
    "The name of the Vertex AI large language model."
    tuned_model_name: Optional[str] = None
    "The name of a tuned model. If provided, model_name is ignored."

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "vertexai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        tuned_model_name = values.get("tuned_model_name")
        model_name = values["model_name"]
        safety_settings = values["safety_settings"]
        is_gemini = is_gemini_model(values["model_name"])
        cls._init_vertexai(values)

        if safety_settings and (not is_gemini or tuned_model_name):
            raise ValueError("Safety settings are only supported for Gemini models")

        if is_codey_model(model_name):
            model_cls = CodeGenerationModel
            preview_model_cls = PreviewCodeGenerationModel
        elif is_gemini:
            model_cls = GenerativeModel
            preview_model_cls = GenerativeModel
        else:
            model_cls = TextGenerationModel
            preview_model_cls = PreviewTextGenerationModel

        if tuned_model_name:
            values["client"] = model_cls.get_tuned_model(tuned_model_name)
            values["client_preview"] = preview_model_cls.get_tuned_model(
                tuned_model_name
            )
        else:
            if is_gemini:
                values["client"] = model_cls(
                    model_name=model_name, safety_settings=safety_settings
                )
                values["client_preview"] = preview_model_cls(
                    model_name=model_name, safety_settings=safety_settings
                )
            else:
                values["client"] = model_cls.from_pretrained(model_name)
                values["client_preview"] = preview_model_cls.from_pretrained(model_name)

        if values["streaming"] and values["n"] > 1:
            raise ValueError("Only one candidate can be generated with streaming!")
        return values

    def _candidate_to_generation(
        self,
        response: Union[Candidate, TextGenerationResponse],
        *,
        stream: bool = False,
        usage_metadata: Optional[Dict] = None,
    ) -> GenerationChunk:
        """Converts a stream response to a generation chunk."""
        generation_info = get_generation_info(
            response,
            self._is_gemini_model,
            stream=stream,
            usage_metadata=usage_metadata,
        )
        try:
            text = response.text
        except AttributeError:
            text = ""
        except ValueError:
            text = ""
        return GenerationChunk(
            text=text,
            generation_info=generation_info,
        )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        should_stream = stream if stream is not None else self.streaming
        params = self._prepare_params(stop=stop, stream=should_stream, **kwargs)
        generations: List[List[Generation]] = []
        for prompt in prompts:
            if should_stream:
                generation = GenerationChunk(text="")
                for chunk in self._stream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    generation += chunk
                generations.append([generation])
            else:
                res = _completion_with_retry(
                    self,
                    [prompt],
                    stream=should_stream,
                    is_gemini=self._is_gemini_model,
                    run_manager=run_manager,
                    **params,
                )
                if self._is_gemini_model:
                    usage_metadata = res.to_dict().get("usage_metadata")
                else:
                    usage_metadata = res.raw_prediction_response.metadata
                generations.append(
                    [
                        self._candidate_to_generation(r, usage_metadata=usage_metadata)
                        for r in res.candidates
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
        params = self._prepare_params(stop=stop, **kwargs)
        generations: List[List[Generation]] = []
        for prompt in prompts:
            res = await _acompletion_with_retry(
                self,
                prompt,
                is_gemini=self._is_gemini_model,
                run_manager=run_manager,
                **params,
            )
            if self._is_gemini_model:
                usage_metadata = res.to_dict().get("usage_metadata")
            else:
                usage_metadata = res.raw_prediction_response.metadata
            generations.append(
                [
                    self._candidate_to_generation(r, usage_metadata=usage_metadata)
                    for r in res.candidates
                ]
            )
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        for stream_resp in _completion_with_retry(
            self,
            [prompt],
            stream=True,
            is_gemini=self._is_gemini_model,
            run_manager=run_manager,
            **params,
        ):
            usage_metadata = None
            if self._is_gemini_model:
                usage_metadata = stream_resp.to_dict().get("usage_metadata")
                stream_resp = stream_resp.candidates[0]
            chunk = self._candidate_to_generation(
                stream_resp, stream=True, usage_metadata=usage_metadata
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
        params = self._prepare_params(stop=stop, stream=True, **kwargs)
        if not self._is_gemini_model:
            raise ValueError("Async streaming is supported only for Gemini family!")
        async for chunk in await _acompletion_with_retry(
            self,
            prompt,
            stream=True,
            is_gemini=self._is_gemini_model,
            run_manager=run_manager,
            **params,
        ):
            usage_metadata = chunk.to_dict().get("usage_metadata")
            chunk = self._candidate_to_generation(
                chunk.candidates[0], stream=True, usage_metadata=usage_metadata
            )
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text, chunk=chunk, verbose=self.verbose
                )
