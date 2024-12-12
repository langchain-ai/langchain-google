"""Standard LangChain interface tests"""

from typing import Dict, List, Literal, Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)
rate_limiter_2_0 = InMemoryRateLimiter(requests_per_second=0.1)


class TestGeminiAI2Standard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-2.0-flash-exp",
            "rate_limiter": rate_limiter_2_0,
        }

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    async def test_structured_output_async(self, model: BaseChatModel) -> None:
        await super().test_structured_output_async(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="investigate")
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)


class TestGeminiAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-1.5-pro-001",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    async def test_structured_output_async(self, model: BaseChatModel) -> None:
        await super().test_structured_output_async(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="Not yet supported")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)

    @property
    def supported_usage_metadata_details(
        self,
    ) -> Dict[
        Literal["invoke", "stream"],
        List[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        return {"invoke": ["cache_read_input"], "stream": ["cache_read_input"]}

    def invoke_with_cache_read_input(self, *, stream: bool = False) -> AIMessage:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001")

        with open(__file__, "r") as f:
            code = f.read() * 40
        cached_content = llm.create_cached_content(
            [
                SystemMessage("you are a good coder"),
                HumanMessage(f"Here is a code file:\n\n```python\n{code}\n```"),
            ],
            ttl=120,
        )
        cached_llm = llm.bind(cached_content=cached_content)

        input_ = "What does the above code do?"

        if stream:
            full = None
            for chunk in cached_llm.stream(input_):
                full = full + chunk if full else chunk  # type: ignore
            return cast(AIMessage, full)
        else:
            return cast(AIMessage, cached_llm.invoke(input_))


# TODO: increase quota on gemini-1.5-pro-001 and test as well
