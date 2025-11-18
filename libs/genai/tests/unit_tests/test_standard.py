from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-2.5-flash"


class TestGeminiAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"GOOGLE_API_KEY": "api_key"},
            self.chat_model_params,
            {"google_api_key": "api_key"},
        )
