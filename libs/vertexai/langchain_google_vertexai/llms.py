from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from google.cloud.aiplatform import telemetry
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM, LangSmithParams
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self
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

from langchain_google_vertexai._base import GoogleModelFamily, _VertexAICommon
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
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

    model_name: str = Field(default="text-bison", alias="model")
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
        tuned_model_name = self.tuned_model_name or None
        safety_settings = self.safety_settings
        self.model_family = GoogleModelFamily(self.model_name)
        is_gemini = is_gemini_model(self.model_family)
        values = {
            "project": self.project,
            "location": self.location,
            "credentials": self.credentials,
            "api_transport": self.api_transport,
            "api_endpoint": self.api_endpoint,
            "default_metadata": self.default_metadata,
        }
        self._init_vertexai(values)

        if safety_settings and (not is_gemini or tuned_model_name):
            raise ValueError("Safety settings are only supported for Gemini models")

        if self.model_family == GoogleModelFamily.CODEY:
            model_cls = CodeGenerationModel
            preview_model_cls = PreviewCodeGenerationModel
        elif is_gemini:
            model_cls = GenerativeModel
            preview_model_cls = GenerativeModel
        else:
            model_cls = TextGenerationModel
            preview_model_cls = PreviewTextGenerationModel

        if tuned_model_name:
            generative_model_name = self.tuned_model_name
        else:
            generative_model_name = self.model_name

        if is_gemini:
            self.client = model_cls(
                model_name=generative_model_name, safety_settings=safety_settings
            )
            self.client_preview = preview_model_cls(
                model_name=generative_model_name, safety_settings=safety_settings
            )
        else:
            if tuned_model_name:
                self.client = model_cls.get_tuned_model(generative_model_name)
                self.client_preview = preview_model_cls.get_tuned_model(
                    generative_model_name
                )
            else:
                self.client = model_cls.from_pretrained(generative_model_name)
                self.client_preview = preview_model_cls.from_pretrained(
                    generative_model_name
                )
        if self.streaming and self.n > 1:
            raise ValueError("Only one candidate can be generated with streaming!")
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
