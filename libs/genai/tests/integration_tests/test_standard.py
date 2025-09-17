"""Standard LangChain interface tests."""

import base64
from typing import Literal

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_genai import ChatGoogleGenerativeAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.25)


class TestGeminiFlashStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "gemini-2.5-flash",
            "rate_limiter": rate_limiter,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Override parent method to use a reliable image URL")
    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Override parent method to use a reliable image URL."""
        if not self.supports_image_inputs:
            pytest.skip("Model does not support image message.")

        # Use a reliable image URL that works with requests
        image_url = "https://picsum.photos/seed/picsum/200/300"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        # OpenAI format, base64 data
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        _ = model.invoke([message])

        # Standard format, base64 data
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/jpeg",
                    "data": image_data,
                },
            ],
        )
        _ = model.invoke([message])

        # Standard format, URL
        if self.supports_image_urls:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image",
                        "source_type": "url",
                        "url": image_url,
                    },
                ],
            )
            _ = model.invoke([message])


class TestGeminiProStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "gemini-2.5-pro",
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(reason="Not yet supported")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @pytest.mark.xfail(
        reason="Investigate: prompt_token_count inconsistent in final chunk."
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
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
