from __future__ import annotations

from concurrent.futures import Executor
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional, Union

import vertexai  # type: ignore[import-untyped]
from google.cloud.aiplatform import telemetry
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
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
    ChatModel as PreviewChatModel,
)
from vertexai.preview.language_models import (
    CodeChatModel as PreviewCodeChatModel,
)
from vertexai.preview.language_models import (
    CodeGenerationModel as PreviewCodeGenerationModel,
)
from vertexai.preview.language_models import (
    TextGenerationModel as PreviewTextGenerationModel,
)

from langchain_google_vertexai._base import (
    _PALM_DEFAULT_MAX_OUTPUT_TOKENS,
    _PALM_DEFAULT_TEMPERATURE,
    _PALM_DEFAULT_TOP_K,
    _PALM_DEFAULT_TOP_P,
)
from langchain_google_vertexai._enums import HarmBlockThreshold, HarmCategory
from langchain_google_vertexai._utils import (
    create_retry_decorator,
    get_generation_info,
    get_user_agent,
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


class _VertexAIBase(BaseModel):
    project: Optional[str] = None
    "The default GCP project to use when making Vertex API calls."
    location: str = "us-central1"
    "The default location to use when making API calls."
    request_parallelism: int = 5
    "The amount of parallelism allowed for requests issued to VertexAI models. "
    "Default is 5."
    max_retries: int = 6
    """The maximum number of retries to make when generating."""
    task_executor: ClassVar[Optional[Executor]] = Field(default=None, exclude=True)
    stop: Optional[List[str]] = None
    "Optional list of stop words to use when generating."
    model_name: Optional[str] = None
    "Underlying model name."


class _VertexAICommon(_VertexAIBase):
    client: Any = None  #: :meta private:
    client_preview: Any = None  #: :meta private:
    model_name: str
    "Underlying model name."
    temperature: Optional[float] = None
    "Sampling temperature, it controls the degree of randomness in token selection."
    max_output_tokens: Optional[int] = None
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: Optional[float] = None
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: Optional[int] = None
    "How the model selects tokens for output, the next token is selected from "
    "among the top-k most probable tokens. Top-k is ignored for Codey models."
    credentials: Any = Field(default=None, exclude=True)
    "The default custom credentials (google.auth.credentials.Credentials) to use "
    "when making API calls. If not provided, credentials will be ascertained from "
    "the environment."
    n: int = 1
    """How many completions to generate for each prompt."""
    streaming: bool = False
    """Whether to stream the results or not."""
    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None
    """The default safety settings to use for all generations. 
    
        For example: 

            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            """  # noqa: E501

    @property
    def _llm_type(self) -> str:
        return "vertexai"

    @property
    def is_codey_model(self) -> bool:
        return is_codey_model(self.model_name)

    @property
    def _is_gemini_model(self) -> bool:
        return is_gemini_model(self.model_name)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Gets the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _default_params(self) -> Dict[str, Any]:
        if self._is_gemini_model:
            default_params = {}
        else:
            default_params = {
                "temperature": _PALM_DEFAULT_TEMPERATURE,
                "max_output_tokens": _PALM_DEFAULT_MAX_OUTPUT_TOKENS,
                "top_p": _PALM_DEFAULT_TOP_P,
                "top_k": _PALM_DEFAULT_TOP_K,
            }
        params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
        }
        if not self.is_codey_model:
            params.update(
                {
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                }
            )
        updated_params = {}
        for param_name, param_value in params.items():
            default_value = default_params.get(param_name)
            if param_value is not None or default_value is not None:
                updated_params[param_name] = (
                    param_value if param_value is not None else default_value
                )
        return updated_params

    @property
    def _user_agent(self) -> str:
        """Gets the User Agent."""
        _, user_agent = get_user_agent(f"{type(self).__name__}_{self.model_name}")
        return user_agent

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        vertexai.init(
            project=values.get("project"),
            location=values.get("location"),
            credentials=values.get("credentials"),
        )
        return None

    def _prepare_params(
        self,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        stop_sequences = stop or self.stop
        params_mapping = {"n": "candidate_count"}
        params = {params_mapping.get(k, k): v for k, v in kwargs.items()}
        params = {**self._default_params, "stop_sequences": stop_sequences, **params}
        if stream or self.streaming:
            params.pop("candidate_count")
        return params

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        is_palm_chat_model = isinstance(
            self.client_preview, PreviewChatModel
        ) or isinstance(self.client_preview, PreviewCodeChatModel)
        if is_palm_chat_model:
            result = self.client_preview.start_chat().count_tokens(text)
        else:
            result = self.client_preview.count_tokens([text])

        return result.total_tokens


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
