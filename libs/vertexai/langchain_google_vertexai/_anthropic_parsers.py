from typing import Any, List, Optional, Type

from langchain_core.messages import AIMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel, ConfigDict


class ToolsOutputParser(BaseGenerationOutputParser):
    first_tool_only: bool = False
    args_only: bool = False
    pydantic_schemas: Optional[List[Type[BaseModel]]] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        if not result or not isinstance(result[0], ChatGeneration):
            return None if self.first_tool_only else []

        message = result[0].message
        tool_calls: List[Any] = []

        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = message.tool_calls
        elif isinstance(message.content, list):
            content: Any = message.content
            tool_calls = _extract_tool_calls(content)

        if self.pydantic_schemas:
            tool_calls = [self._pydantic_parse(tc) for tc in tool_calls]
        elif self.args_only:
            tool_calls = [tc["args"] for tc in tool_calls]

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        else:
            return [tool_call for tool_call in tool_calls]

    def _pydantic_parse(self, tool_call: dict) -> BaseModel:
        cls_ = {schema.__name__: schema for schema in self.pydantic_schemas or []}[
            tool_call["name"]
        ]
        return cls_(**tool_call["args"])


def _extract_tool_calls(content: List[dict]) -> List[ToolCall]:
    tool_calls = []
    for block in content:
        if block["type"] == "tool_use":
            tool_calls.append(
                tool_call(name=block["name"], args=block["input"], id=block["id"])
            )
    return tool_calls
