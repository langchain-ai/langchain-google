"""Standard LangChain interface tests"""

import json
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests
from langchain_standard_tests.unit_tests.chat_models import my_adder_tool

from langchain_google_vertexai import ChatVertexAI


class TestGeminiAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {"model_name": "gemini-1.0-pro-001"}

    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        # parent implementation uses tool_choice='any':
        # model_with_tools = model.bind_tools([my_adder_tool], tool_choice="any")

        # gemini 1 doesn't support tool_choice="any":
        model_with_tools = model.bind_tools([my_adder_tool])

        function_name = "my_adder_tool"
        function_args = {"a": 1, "b": 2}
        function_result = json.dumps({"result": 3})

        messages_string_content = [
            HumanMessage(content="What is 1 + 2"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": function_name,
                        "args": function_args,
                        "id": "abc123",
                    },
                ],
            ),
            ToolMessage(
                content=function_result,
                name=function_name,
                tool_call_id="abc123",
            ),
            AIMessage(content=function_result),
            HumanMessage(content="What is 3 + 4"),
        ]
        result_string_content = model_with_tools.invoke(messages_string_content)
        assert isinstance(result_string_content, AIMessage)


class TestGemini_15_AIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatVertexAI

    @property
    def chat_model_params(self) -> dict:
        return {"model_name": "gemini-1.5-pro-001"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True
