from typing import Type
from unittest.mock import patch

from google.cloud.aiplatform_v1beta1.types import (
    GenerateContentResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_vertexai import ChatVertexAI


class TestGemini_AIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        with patch(
            "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
        ) as mock_prediction_service:
            response = GenerateContentResponse(candidates=[])
            mock_prediction_service.return_value.generate_content.return_value = (
                response
            )

            return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {"model_name": "gemini-2.5-pro", "project": "test-proj"}
