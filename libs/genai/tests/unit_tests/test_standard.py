from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_genai import ChatGoogleGenerativeAI


class TestGeminiAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gemini-2.5-flash"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            self.chat_model_params,
            {"google_api_key": "api_key"},
        )


class TestGemini15AIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gemini-2.5-flash"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            self.chat_model_params,
            {"google_api_key": "api_key"},
        )
