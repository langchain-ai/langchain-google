from unittest.mock import patch

import pytest
from google.cloud.aiplatform_v1beta1.types import (
    GenerateContentResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_google_vertexai import ChatVertexAI

# Suppress warnings about parameters not supported by ChatVertexAI
# The standard test suite passes generic parameters like 'timeout' and 'api_key'
# that are common across providers, but ChatVertexAI uses Google Cloud auth
# (service accounts, ADC, workload identity) and handles timeouts via client config
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*timeout.*not default parameter.*:UserWarning",
    "ignore:.*api_key.*not default parameter.*:UserWarning",
)


class TestGemini_AIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
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
