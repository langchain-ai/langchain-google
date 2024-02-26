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
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain_google_vertexai._base import _BaseVertexAIModelGarden
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


class _GemmaBase(BaseModel):
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    temperature: Optional[float] = None
    """The temperature to use for sampling."""
    top_p: Optional[float] = None
    """The top-p value to use for sampling."""
    top_k: Optional[int] = None
    """The top-k value to use for sampling."""

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _get_params(self, **kwargs) -> Dict[str, Any]:
        return {k: kwargs.get(k, v) for k, v in self._default_params.items()}


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


class GemmaChatVertexAIModelGarden(_GemmaBase, _BaseVertexAIModelGarden, BaseChatModel):
    allowed_model_args: Optional[List[str]] = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
    ]

    @property
    def _llm_type(self) -> str:
        return "gemma_vertexai_model_garden"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {"max_length": self.max_tokens}
        return {k: v for k, v in params.items() if v is not None}

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
        generations = [
            ChatGeneration(
                message=AIMessage(content=output.predictions[0]),
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
        output = await self.async_client.predict_(
            endpoint=self.endpoint_path, instances=[request]
        )
        generations = [
            ChatGeneration(
                message=AIMessage(content=output.predictions[0]),
            )
        ]
        return ChatResult(generations=generations)


class _GemmaLocalKaggleBase(_GemmaBase):
    """Local gemma model."""

    client: Any = None  #: :meta private:
    keras_backend: str = "jax"
    model_name: str = "gemma_2b_en"
    """Gemma model name."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        try:
            os.environ["KERAS_BACKEND"] = values["keras_backend"]
            from keras_nlp.models import GemmaCausalLM  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import GemmaCausalLM library. "
                "Please install the GemmaCausalLM library to "
                "use this  model: pip install keras-nlp keras>=3 kaggle"
            )

        values["client"] = GemmaCausalLM.from_preset(values["model_name"])
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling gemma."""
        params = {"max_length": self.max_tokens}
        return {k: v for k, v in params.items() if v is not None}


class GemmaLocalKaggle(_GemmaLocalKaggleBase, BaseLLM):
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        params = {"max_length": self.max_tokens} if self.max_tokens else {}
        results = self.client.generate(prompts, **params)
        results = results if isinstance(results, str) else [results]
        return LLMResult(generations=[[Generation(text=result)] for result in results])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_kaggle"


class GemmaChatLocalKaggle(_GemmaLocalKaggleBase, BaseChatModel):
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = {"max_length": self.max_tokens} if self.max_tokens else {}
        prompt = gemma_messages_to_prompt(messages)
        output = self.client.generate(prompt, **params)
        generation = ChatGeneration(message=AIMessage(content=output))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gemma_local_chat_kaggle"
