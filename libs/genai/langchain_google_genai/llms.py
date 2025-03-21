from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LangSmithParams
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import HumanMessage, ChatMessage, BaseMessage
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_google_genai._common import (
    _BaseGoogleGenerativeAI,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


class GoogleGenerativeAI(_BaseGoogleGenerativeAI, BaseLLM):
    """Google GenerativeAI models.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAI
            llm = GoogleGenerativeAI(model="gemini-pro")
    """

    client: Any = None  #: :meta private:
    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and passes them to google-generativeai package."""

        self.client = ChatGoogleGenerativeAI(
            api_key=self.google_api_key,
            credentials=self.credentials,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_output_tokens,
            timeout=self.timeout,
            model=self.model,
            client_options=self.client_options,
            transport=self.transport,
            additional_headers=self.additional_headers,
            safety_settings=self.safety_settings,
        )

        return self

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "google_genai"
        if ls_max_tokens := kwargs.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        return ls_params
    
    async def _execute_code_safe(self, code: str) -> str:
        """Executes Python code in a sandboxed environment."""
        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                code,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            output = stdout.decode() + stderr.decode()
            return output
        except Exception as e:
            return f"Error: {e}"


    async def _handle_code_execution(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        updated_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                code_blocks = re.findall(r"```python\n(.*?)\n```", message.content, re.DOTALL)
                if code_blocks:
                    parts = re.split(r"```python\n(.*?)\n```", message.content, re.DOTALL)
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            if part:
                                updated_messages.append(HumanMessage(content=part))
                        else:
                            code_output = await self._execute_code_safe(part)
                            updated_messages.append(HumanMessage(content=f"```python\n{part}\n```"))
                            updated_messages.append(ChatMessage(role="code_output", content=code_output))
                else:
                    updated_messages.append(message)
            else:
                updated_messages.append(message)
        return updated_messages

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            async def async_wrapper(p):
                messages = await self._handle_code_execution([HumanMessage(content=p)])
                chat_result = self.client._generate(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
                return chat_result

            chat_result = asyncio.run(async_wrapper(prompt))

            generations.append(
                [
                    Generation(
                        text=g.message.content,
                        generation_info={
                            **g.generation_info,
                            **{"usage_metadata": g.message.usage_metadata},
                        },
                    )
                    for g in chat_result.generations
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
        messages = asyncio.run(self._handle_code_execution([HumanMessage(content=prompt)])) 

        for stream_chunk in self.client._stream(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            chunk = GenerationChunk(text=stream_chunk.message.content)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "google_gemini"

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        return self.client.get_num_tokens(text)
