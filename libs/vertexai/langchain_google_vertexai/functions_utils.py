from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
)

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.function_calling import FunctionDescription
from langchain_core.utils.json_schema import dereference_refs
from vertexai.generative_models import FunctionDeclaration  # type: ignore
from vertexai.generative_models import Tool as VertexTool

# FIXME: vertexai is not exporting ToolConfig
from vertexai.generative_models._generative_models import ToolConfig  # type: ignore


def _format_pydantic_to_vertex_function(
    pydantic_model: Type[BaseModel],
) -> FunctionDescription:
    schema = pydantic_model.schema()

    return {
        "name": schema["title"],
        "description": schema.get("description", ""),
        "parameters": _get_parameters_from_schema(schema=schema),
    }


def _format_base_tool_to_vertex_function(tool: BaseTool) -> FunctionDescription:
    "Format tool into the Vertex function API."
    if tool.args_schema:
        schema = tool.args_schema.schema()

        return {
            "name": tool.name or schema["title"],
            "description": tool.description or schema["description"],
            "parameters": _get_parameters_from_schema(schema=schema),
        }
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "properties": {
                    "__arg1": {"type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            },
        }


def _format_to_vertex_function_dict(
    tool: Union[BaseTool, Type[BaseModel], dict, Callable, FunctionDeclaration],
) -> FunctionDescription:
    "Format tool into the Vertex function declaration."
    if isinstance(tool, BaseTool):
        return _format_base_tool_to_vertex_function(tool)
    elif isinstance(tool, type) and issubclass(tool, BaseModel):
        return _format_pydantic_to_vertex_function(tool)
    elif isinstance(tool, dict):
        return {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": _get_parameters_from_schema(tool["parameters"]),
        }
    elif isinstance(tool, FunctionDeclaration):
        tool_dict = tool.to_dict()
        return FunctionDescription(
            name=tool_dict["name"],
            description=tool_dict["description"],
            parameters=tool_dict["parameters"],
        )
    elif callable(tool):
        return _format_base_tool_to_vertex_function(callable_as_lc_tool()(tool))
    else:
        raise ValueError(f"Unsupported tool call type {tool}")


_FunctionDeclarationLike = Union[
    BaseTool, Type[BaseModel], dict, Callable, FunctionDeclaration
]


class _VertexToolDict(TypedDict):
    function_declarations: List[FunctionDescription]


def _format_functions_to_vertex_tool_dict(
    functions: List[_FunctionDeclarationLike],
) -> _VertexToolDict:
    "Format tools into the Vertex Tool instance."
    function_declarations = [_format_to_vertex_function_dict(fn) for fn in functions]
    return _VertexToolDict(function_declarations=function_declarations)


def _format_to_vertex_tool(
    tool: Union[VertexTool, _VertexToolDict, List[_FunctionDeclarationLike]],
) -> VertexTool:
    if isinstance(tool, VertexTool):
        return tool
    elif isinstance(tool, (list, dict)):
        tool = (
            _format_functions_to_vertex_tool_dict(tool)
            if isinstance(tool, list)
            else tool
        )
        return VertexTool(
            function_declarations=[
                FunctionDeclaration(**fd) for fd in tool["function_declarations"]
            ]
        )
    else:
        raise ValueError(f"Unexpected tool value:\n\n{tool=}")


def _format_tool_config(tool_config: _ToolConfigDict) -> Union[ToolConfig, None]:
    if "function_calling_config" not in tool_config:
        raise ValueError(
            "Invalid ToolConfig, missing 'function_calling_config' key. Received:\n\n"
            f"{tool_config=}"
        )
    return ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            **tool_config["function_calling_config"]
        )
    )


class ParametersSchema(BaseModel):
    """
    This is a schema of currently supported definitions in function calling.
    We need explicitly exclude `definitions` field
    as it is not currently supported.

    All other fields will be passed through (as extra fields are allowed)
    and intercepted on `google.cloud.aiplatform` level
    """

    definitions: Optional[Any] = Field(exclude=True)
    items: Optional["ParametersSchema"]
    properties: Optional[Dict[str, "ParametersSchema"]]

    class Config:
        extra = "allow"


def _get_parameters_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Given a schema, format the parameters key to match VertexAI
    expected input.

    Args:
        schema: Dictionary that must have the following keys.

    Returns:
        Dictionary with the formatted parameters.
    """

    dereferenced_schema = dereference_refs(schema)
    model = ParametersSchema.parse_obj(dereferenced_schema)

    return model.dict(exclude_unset=True)


class PydanticFunctionsOutputParser(BaseOutputParser):
    """Parse an output as a pydantic object.

    This parser is used to parse the output of a ChatModel that uses
    Google Vertex function format to invoke functions.

    The parser extracts the function call invocation and matches
    them to the pydantic schema provided.

    An exception will be raised if the function call does not match
    the provided schema.

    Example:

        ... code-block:: python

            message = AIMessage(
                content="This is a test message",
                additional_kwargs={
                    "function_call": {
                        "name": "cookie",
                        "arguments": json.dumps({"name": "value", "age": 10}),
                    }
                },
            )
            chat_generation = ChatGeneration(message=message)

            class Cookie(BaseModel):
                name: str
                age: int

            class Dog(BaseModel):
                species: str

            # Full output
            parser = PydanticOutputFunctionsParser(
                pydantic_schema={"cookie": Cookie, "dog": Dog}
            )
            result = parser.parse_result([chat_generation])
    """

    pydantic_schema: Union[Type[BaseModel], Dict[str, Type[BaseModel]]]

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> BaseModel:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        function_call = message.additional_kwargs.get("function_call", {})
        if function_call:
            function_name = function_call["name"]
            tool_input = function_call.get("arguments", {})
            if isinstance(self.pydantic_schema, dict):
                schema = self.pydantic_schema[function_name]
            else:
                schema = self.pydantic_schema
            return schema(**json.loads(tool_input))
        else:
            raise OutputParserException(f"Could not parse function call: {message}")

    def parse(self, text: str) -> BaseModel:
        raise ValueError("Can only parse messages")


class _FunctionCallingConfigDict(TypedDict):
    mode: ToolConfig.FunctionCallingConfig.Mode
    allowed_function_names: Optional[List[str]]


class _ToolConfigDict(TypedDict):
    function_calling_config: _FunctionCallingConfigDict


_ToolChoiceType = Union[
    dict, List[str], str, Literal["auto", "none", "any"], Literal[True]
]


def _tool_choice_to_tool_config(
    tool_choice: _ToolChoiceType,
    all_names: List[str],
) -> _ToolConfigDict:
    allowed_function_names: Optional[List[str]] = None
    if tool_choice is True or tool_choice == "any":
        mode = ToolConfig.FunctionCallingConfig.Mode.ANY
        allowed_function_names = all_names
    elif tool_choice == "auto":
        mode = ToolConfig.FunctionCallingConfig.Mode.AUTO
    elif tool_choice == "none":
        mode = ToolConfig.FunctionCallingConfig.Mode.NONE
    elif isinstance(tool_choice, str):
        mode = ToolConfig.FunctionCallingConfig.Mode.ANY
        allowed_function_names = [tool_choice]
    elif isinstance(tool_choice, list):
        mode = ToolConfig.FunctionCallingConfig.Mode.ANY
        allowed_function_names = tool_choice
    elif isinstance(tool_choice, dict):
        if "mode" in tool_choice:
            mode = tool_choice["mode"]
            allowed_function_names = tool_choice.get("allowed_function_names")
        elif "function_calling_config" in tool_choice:
            mode = tool_choice["function_calling_config"]["mode"]
            allowed_function_names = tool_choice["function_calling_config"].get(
                "allowed_function_names"
            )
        else:
            raise ValueError(
                f"Unrecognized tool choice format:\n\n{tool_choice=}\n\nShould match "
                f"VertexAI ToolConfig or FunctionCallingConfig format."
            )
    else:
        raise ValueError(f"Unrecognized tool choice format:\n\n{tool_choice=}")
    return _ToolConfigDict(
        function_calling_config=_FunctionCallingConfigDict(
            mode=mode,
            allowed_function_names=allowed_function_names,
        )
    )
