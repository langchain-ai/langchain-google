"""Standard LangChain interface tests"""

import base64
from typing import Type

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_google_vertexai import ChatVertexAI

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


@pytest.mark.first
class TestGemini2AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-2.0-flash-001",
            "rate_limiter": rate_limiter,
            "temperature": 0,
            "api_transport": None,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Override parent method to use a reliable image URL."""
        if not self.supports_image_inputs:
            pytest.skip("Model does not support image message.")

        # Use a reliable image URL that works with requests
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
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


class TestGemini_15_AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "gemini-2.5-pro",
            "rate_limiter": rate_limiter,
            "temperature": 0,
            "api_transport": None,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    def test_image_inputs(self, model: BaseChatModel) -> None:
        """Override parent method to use a reliable image URL."""
        if not self.supports_image_inputs:
            pytest.skip("Model does not support image message.")

        # Use a reliable image URL that works with requests
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
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
