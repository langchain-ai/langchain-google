"""Standard LangChain interface tests"""

from typing import Dict, List, Literal, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)


class TestGeminiAI2Standard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "models/gemini-2.0-flash-001",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(
        reason="Likely a bug in genai: prompt_token_count inconsistent in final chunk."
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> None:
        await super().test_structured_output_async(model, schema_type)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason="investigate")
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)

    @pytest.mark.xfail(reason=("investigate"))
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)


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
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> None:
        await super().test_structured_output_async(model, schema_type)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        super().test_structured_output(model, schema_type)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="with_structured_output with JSON schema not supported.")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason="Not yet supported")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

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
        return {"invoke": [], "stream": []}


# TODO: increase quota on gemini-1.5-pro-001 and test as well
