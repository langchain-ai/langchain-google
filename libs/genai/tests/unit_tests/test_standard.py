from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_genai import ChatGoogleGenerativeAI


class TestGeminiAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "models/gemini-1.0-pro-001"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            self.chat_model_params,
            {"google_api_key": "api_key"},
        )


class TestGemini_15_AIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "models/gemini-1.5-pro-001"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            self.chat_model_params,
            {"google_api_key": "api_key"},
        )
