import os
from typing import Any, Dict, List, Optional, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    Generation,
    LLMResult,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_google_vertexai._base import _BaseVertexAIModelGarden
from langchain_google_vertexai._utils import enforce_stop_tokens
from langchain_google_vertexai.model_garden import VertexAIModelGarden

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"


def gemma_messages_to_prompt(history: List[BaseMessage]) -> str:
    """Converts a list of messages to a chat prompt for Gemma."""
    messages: List[str] = []
    if len(messages) == 1:
        content = cast(str, history[0].content)
        if isinstance(history[0], SystemMessage):
            raise ValueError("Gemma currently doesn't support system message!")
        return content
    for message in history:
        content = cast(str, message.content)
        if isinstance(message, SystemMessage):
            raise ValueError("Gemma currently doesn't support system message!")
        elif isinstance(message, AIMessage):
            messages.append(MODEL_CHAT_TEMPLATE.format(prompt=content))
        elif isinstance(message, HumanMessage):
            messages.append(USER_CHAT_TEMPLATE.format(prompt=content))
        else:
            raise ValueError(f"Unexpected message with type {type(message)}")
    messages.append("<start_of_turn>model\n")
    return "".join(messages)


def _parse_gemma_chat_response(response: str) -> str:
    """Removes chat history from the response."""
    pattern = "<start_of_turn>model\n"
    pos = response.rfind(pattern)
    if pos == -1:
        return response
    text = response[(pos + len(pattern)) :]
    pos = text.find("<start_of_turn>user\n")
    if pos > 0:
        return text[:pos]
    return text


class _GemmaBase(BaseModel):
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    temperature: Optional[float] = None
    """The temperature to use for sampling."""
    top_p: Optional[float] = None
    """The top-p value to use for sampling."""
    top_k: Optional[int] = None
    """The top-k value to use for sampling."""

    model_config = ConfigDict(protected_namespaces=())

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        return {k: v for k, v in params.items()}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        params = {k: kwargs.get(k, v) for k, v in self._default_params.items()}
        return {k: v for k, v in params.items() if v is not None}


class GemmaVertexAIModelGarden(VertexAIModelGarden):
    allowed_model_args: Optional[List[str]] = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
    ]

    @property
    def _llm_type(self) -> str:
        return "gemma_vertexai_model_garden"

    # Needed so that mypy doesn't flag missing aliased init args.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class GemmaChatVertexAIModelGarden(_GemmaBase, _BaseVertexAIModelGarden, BaseChatModel):
    allowed_model_args: Optional[List[str]] = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_length",
    ]
    parse_response: bool = False
    """Whether to post-process the chat response and clean repeations """
    """or multi-turn statements."""

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg."""
        if model_name:
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=(),
    )

    @property
    def _llm_type(self) -> str:
        return "gemma_vertexai_model_garden"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        # support both Gemma 1B and 2B
        params = super()._default_params
        params["max_length"] = self.max_tokens
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._get_params(**kwargs)
        request["prompt"] = gemma_messages_to_prompt(messages)
        output = self.client.predict(endpoint=self.endpoint_path, instances=[request])
        text = output.predictions[0]
        if self.parse_response or kwargs.get("parse_response"):
            text = _parse_gemma_chat_response(text)
        if stop:
            text = enforce_stop_tokens(text, stop)
        generations = [
            ChatGeneration(
                message=AIMessage(content=text),
            )
        ]
        return ChatResult(generations=generations)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        request = self._get_params(**kwargs)
        request["prompt"] = gemma_messages_to_prompt(messages)
        output = await self.async_client.predict(
            endpoint=self.endpoint_path, instances=[request]
        )
        text = output.predictions[0]
        if self.parse_response or kwargs.get("parse_response"):
            text = _parse_gemma_chat_response(text)
        if stop:
            text = enforce_stop_tokens(text, stop)
        generations = [
            ChatGeneration(
                message=AIMessage(content=text),
            )
        ]
        return ChatResult(generations=generations)


class _GemmaLocalKaggleBase(_GemmaBase):
    """Local gemma model loaded from Kaggle."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    keras_backend: str = "jax"
    model_name: str = Field(default="gemma_2b_en", alias="model")
    """Gemma model name."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg."""
        if model_name:
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that llama-cpp-python library is installed."""
        try:
            os.environ["KERAS_BACKEND"] = self.keras_backend
            from keras_nlp.models import GemmaCausalLM  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import GemmaCausalLM library. "
                "Please install the GemmaCausalLM library to "
                "use this  model: pip install keras-nlp keras>=3 kaggle"
            )

        self.client = GemmaCausalLM.from_preset(self.model_name)
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {"max_length": self.max_tokens}
        return {k: v for k, v in params.items() if v is not None}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        mapping = {"max_tokens": "max_length"}
        params = {mapping[k]: v for k, v in kwargs.items() if k in mapping}
        return {**self._default_params, **params}


class GemmaLocalKaggle(_GemmaLocalKaggleBase, BaseLLM):
    """Local gemma chat model loaded from Kaggle."""

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Only needed for typing."""
        if model_name:
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        params = self._get_params(**kwargs)
        results = self.client.generate(prompts, **params)
        results = [results] if isinstance(results, str) else results
        if stop:
            results = [enforce_stop_tokens(text, stop) for text in results]
        return LLMResult(generations=[[Generation(text=result)] for result in results])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_kaggle"


class GemmaChatLocalKaggle(_GemmaLocalKaggleBase, BaseChatModel):
    parse_response: bool = False
    """Whether to post-process the chat response and clean repeations """
    """or multi-turn statements."""

    def __init__(self, *, model_name: Optional[str] = None, **kwargs: Any) -> None:
        """Needed for mypy typing to recognize model_name as a valid arg."""
        if model_name:
            kwargs["model_name"] = model_name
        super().__init__(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._get_params(**kwargs)
        prompt = gemma_messages_to_prompt(messages)
        text = self.client.generate(prompt, **params)
        if self.parse_response or kwargs.get("parse_response"):
            text = _parse_gemma_chat_response(text)
        if stop:
            text = enforce_stop_tokens(text, stop)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_chat_kaggle"


class _GemmaLocalHFBase(_GemmaBase):
    """Local gemma model loaded from HuggingFace."""

    tokenizer: Any = None  #: :meta private:
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    hf_access_token: str
    cache_dir: Optional[str] = None
    model_name: str = Field(default="google/gemma-2b", alias="model")
    """Gemma model name."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that llama-cpp-python library is installed."""
        try:
            from transformers import AutoTokenizer, GemmaForCausalLM  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import GemmaForCausalLM library. "
                "Please install the GemmaForCausalLM library to "
                "use this  model: pip install transformers>=4.38.1"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.hf_access_token
        )
        self.client = GemmaForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_access_token,
            cache_dir=self.cache_dir,
        )
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {"max_length": self.max_tokens}
        return {k: v for k, v in params.items() if v is not None}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        mapping = {"max_tokens": "max_length"}
        params = {mapping[k]: v for k, v in kwargs.items() if k in mapping}
        return {**self._default_params, **params}

    def _run(self, prompt: str, **kwargs: Any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        params = self._get_params(**kwargs)
        generate_ids = self.client.generate(inputs.input_ids, **params)
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class GemmaLocalHF(_GemmaLocalHFBase, BaseLLM):
    """Local gemma model loaded from HuggingFace."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        results = [self._run(prompt, **kwargs) for prompt in prompts]
        if stop:
            results = [enforce_stop_tokens(text, stop) for text in results]
        return LLMResult(generations=[[Generation(text=text)] for text in results])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_hf"


class GemmaChatLocalHF(_GemmaLocalHFBase, BaseChatModel):
    parse_response: bool = False
    """Whether to post-process the chat response and clean repeations """
    """or multi-turn statements."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = gemma_messages_to_prompt(messages)
        text = self._run(prompt, **kwargs)
        if self.parse_response or kwargs.get("parse_response"):
            text = _parse_gemma_chat_response(text)
        if stop:
            text = enforce_stop_tokens(text, stop)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_chat_hf"
