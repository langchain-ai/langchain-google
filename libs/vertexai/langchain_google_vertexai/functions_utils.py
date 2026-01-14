from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    TypedDict,
    Union,
    cast,
)

import google.cloud.aiplatform_v1beta1.types as gapic
import vertexai.generative_models as vertexai  # TODO: migrate to google-genai
from google.cloud.aiplatform_v1beta1.types import (
    ToolConfig as GapicToolConfig,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.function_calling import (
    FunctionDescription,
    convert_to_openai_tool,
)
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

_FunctionDeclarationLike = Union[
    BaseTool,
    type[BaseModel],
    FunctionDescription,
    Callable,
    vertexai.FunctionDeclaration,
    dict[str, Any],
]
_GoogleSearchRetrievalLike = Union[
    gapic.GoogleSearchRetrieval,
    dict[str, Any],
]
_GoogleSearchLike = Union[gapic.Tool.GoogleSearch, dict[str, Any]]
_RetrievalLike = Union[gapic.Retrieval, dict[str, Any]]
_CodeExecutionLike = Union[gapic.Tool.CodeExecution, dict[str, Any]]


class _ToolDictLike(TypedDict):
    function_declarations: list[_FunctionDeclarationLike] | None
    google_search_retrieval: _GoogleSearchRetrievalLike | None
    google_search: _GoogleSearchLike | None
    retrieval: _RetrievalLike | None
    code_execution: NotRequired[_CodeExecutionLike]


_ToolType = Union[gapic.Tool, vertexai.Tool, _ToolDictLike, _FunctionDeclarationLike]
_ToolsType = Sequence[_ToolType]

_ALLOWED_SCHEMA_FIELDS = []
_ALLOWED_SCHEMA_FIELDS.extend([f.name for f in gapic.Schema()._pb.DESCRIPTOR.fields])
_ALLOWED_SCHEMA_FIELDS.extend(
    list(gapic.Schema.to_dict(gapic.Schema(), preserving_proto_field_name=False).keys())
)
_ALLOWED_SCHEMA_FIELDS_SET = set(_ALLOWED_SCHEMA_FIELDS)


def _format_json_schema_to_gapic_v1(schema: dict[str, Any]) -> dict[str, Any]:
    """Format a JSON schema from a Pydantic V1 `BaseModel` to gapic."""
    converted_schema: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "definitions":
            continue
        if key == "items":
            converted_schema["items"] = _format_json_schema_to_gapic_v1(value)
        elif key == "properties":
            if "properties" not in converted_schema:
                converted_schema["properties"] = {}
            for pkey, pvalue in value.items():
                converted_schema["properties"][pkey] = _format_json_schema_to_gapic_v1(
                    pvalue
                )
            continue
        elif key in ["type", "_type"]:
            converted_schema["type"] = str(value).upper()
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic_v1(value[0])
        elif key == "anyOf":
            converted_schema["anyOf"] = [
                _format_json_schema_to_gapic_v1(anyOf_type) for anyOf_type in value
            ]
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _format_json_schema_to_gapic(
    schema: dict[str, Any],
    parent_key: str | None = None,
    required_fields: list | None = None,
) -> dict[str, Any]:
    """Format a JSON schema from a Pydantic V2 `BaseModel` to gapic."""
    converted_schema: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "$defs":
            continue
        if key == "items":
            converted_schema["items"] = _format_json_schema_to_gapic(
                value, parent_key, required_fields
            )
        elif key == "properties":
            if "properties" not in converted_schema:
                converted_schema["properties"] = {}
            for pkey, pvalue in value.items():
                converted_schema["properties"][pkey] = _format_json_schema_to_gapic(
                    pvalue, pkey, schema.get("required", [])
                )
            continue
        elif key in ["type", "_type"]:
            converted_schema["type"] = str(value).upper()
        elif key == "allOf":
            if len(value) > 1:
                logger.warning(
                    "Only first value for 'allOf' key is supported. "
                    f"Got {len(value)}, ignoring other than first value!"
                )
            return _format_json_schema_to_gapic(value[0], parent_key, required_fields)
        elif key == "anyOf":
            if any(v.get("type") == "null" for v in value):
                non_null_type = next(v for v in value if v.get("type") != "null")
                converted_schema.update(
                    _format_json_schema_to_gapic(
                        non_null_type, parent_key, required_fields
                    )
                )
                # Remove the field from required if it exists
                if required_fields and parent_key in required_fields:
                    required_fields.remove(parent_key)
                continue
            converted_schema["anyOf"] = [
                _format_json_schema_to_gapic(
                    anyOf_type, "anyOf", schema.get("required", [])
                )
                for anyOf_type in value
            ]
        elif key not in _ALLOWED_SCHEMA_FIELDS_SET:
            logger.warning(f"Key '{key}' is not supported in schema, ignoring")
        else:
            converted_schema[key] = value
    return converted_schema


def _dict_to_gapic_schema(
    schema: dict[str, Any], pydantic_version: str = "v1"
) -> gapic.Schema:
    # Resolve refs in schema because $refs and $defs are not supported
    # by the Gemini API.
    dereferenced_schema = dereference_refs(schema)

    if pydantic_version == "v1":
        formatted_schema = _format_json_schema_to_gapic_v1(dereferenced_schema)
    else:
        formatted_schema = _format_json_schema_to_gapic(dereferenced_schema)
    json_schema = json.dumps(formatted_schema)
    return gapic.Schema.from_json(json_schema)


def _format_base_tool_to_function_declaration(
    tool: BaseTool,
) -> gapic.FunctionDeclaration:
    """Format tool into the Vertex function API."""
    if not tool.args_schema:
        return gapic.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=gapic.Schema(
                type=gapic.Type.OBJECT,
                properties={
                    "__arg1": gapic.Schema(type=gapic.Type.STRING),
                },
                required=["__arg1"],
            ),
        )

    if hasattr(tool.args_schema, "model_json_schema"):
        schema = tool.args_schema.model_json_schema(mode="serialization")
        pydantic_version = "v2"
    else:
        schema = tool.args_schema.schema()  # type: ignore[attr-defined]
        pydantic_version = "v1"

    parameters = _dict_to_gapic_schema(schema, pydantic_version=pydantic_version)

    return gapic.FunctionDeclaration(
        name=tool.name or schema.get("title"),
        description=tool.description or schema.get("description"),
        parameters=parameters,
    )


def _format_pydantic_to_function_declaration(
    pydantic_model: type[BaseModel],
) -> gapic.FunctionDeclaration:
    if hasattr(pydantic_model, "model_json_schema"):
        schema = pydantic_model.model_json_schema(mode="serialization")
        pydantic_version = "v2"
    else:
        schema = pydantic_model.schema()
        pydantic_version = "v1"

    return gapic.FunctionDeclaration(
        name=schema["title"],
        description=schema.get("description", ""),
        parameters=_dict_to_gapic_schema(schema, pydantic_version=pydantic_version),
    )


def _format_dict_to_function_declaration(
    tool: FunctionDescription | dict[str, Any],
) -> gapic.FunctionDeclaration:
    pydantic_version_v2 = False

    # Ensure we send "anyOf" parameters through pydantic v2 schema parsing
    def _check_v2(parameters) -> bool:
        properties = parameters.get("properties", {}).values()
        for property in properties:
            if "anyOf" in property:
                return True
            if "parameters" in property:
                if _check_v2(property["parameters"]):
                    return True
            if "items" in property and _check_v2(property["items"]):
                return True
        return False

    if isinstance(tool, dict):
        pydantic_version_v2 = _check_v2(tool.get("parameters", {}))
    if pydantic_version_v2:
        parameters = _dict_to_gapic_schema(
            tool.get("parameters", {}), pydantic_version="v2"
        )
    else:
        parameters = _dict_to_gapic_schema(tool.get("parameters", {}))

    return gapic.FunctionDeclaration(
        name=tool.get("name"),
        description=tool.get("description"),
        parameters=parameters,
    )


def _format_vertex_to_function_declaration(
    tool: vertexai.FunctionDeclaration,
) -> gapic.FunctionDeclaration:
    tool_dict = tool.to_dict()
    return _format_dict_to_function_declaration(tool_dict)


def _format_to_gapic_function_declaration(
    tool: _FunctionDeclarationLike,
) -> gapic.FunctionDeclaration:
    """Format tool into the Vertex function declaration."""
    if isinstance(tool, BaseTool):
        return _format_base_tool_to_function_declaration(tool)
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        return _format_pydantic_to_function_declaration(tool)
    if callable(tool) and not (
        isinstance(tool, type) and hasattr(tool, "__annotations__")
    ):
        return _format_base_tool_to_function_declaration(callable_as_lc_tool()(tool))
    if isinstance(tool, vertexai.FunctionDeclaration):
        return _format_vertex_to_function_declaration(tool)
    if isinstance(tool, dict) or (
        isinstance(tool, type) and hasattr(tool, "__annotations__")
    ):
        # this could come from
        # 'langchain_core.utils.function_calling.convert_to_openai_tool'
        function = convert_to_openai_tool(cast("dict", tool))["function"]
        return _format_dict_to_function_declaration(
            cast("FunctionDescription", function)
        )
    msg = f"Unsupported tool call type {tool}"
    raise ValueError(msg)


def _format_to_gapic_tool(tools: _ToolsType) -> gapic.Tool:
    gapic_tool = gapic.Tool()
    for tool in tools:
        if any(f in gapic_tool for f in ["google_search_retrieval", "retrieval"]):
            msg = (
                "Providing multiple retrieval, google_search_retrieval"
                " or mixing with function_declarations is not supported"
            )
            raise ValueError(msg)
        if isinstance(tool, (gapic.Tool, vertexai.Tool)):
            rt: gapic.Tool = (
                tool if isinstance(tool, gapic.Tool) else tool._raw_tool  # type: ignore
            )
            if "retrieval" in rt:
                gapic_tool.retrieval = rt.retrieval
            if "google_search_retrieval" in rt:
                gapic_tool.google_search_retrieval = rt.google_search_retrieval
            if "function_declarations" in rt:
                gapic_tool.function_declarations.extend(rt.function_declarations)
            if "google_search" in rt:
                gapic_tool.google_search = rt.google_search
            if "code_execution" in rt:
                gapic_tool.code_execution = rt.code_execution
        elif isinstance(tool, dict):
            # not _ToolDictLike
            if not any(
                f in tool
                for f in [
                    "function_declarations",
                    "google_search_retrieval",
                    "google_search",
                    "retrieval",
                    "code_execution",
                ]
            ):
                # Type ignore: tool is dict but mypy can't verify it's valid
                # _FunctionDeclarationLike. Runtime handles invalid types properly
                fd = _format_to_gapic_function_declaration(tool)  # type: ignore
                gapic_tool.function_declarations.append(fd)
                continue
            # _ToolDictLike
            tool = cast("_ToolDictLike", tool)
            if "function_declarations" in tool:
                function_declarations = tool["function_declarations"]
                if not isinstance(tool["function_declarations"], list):
                    msg = (
                        "function_declarations should be a list"
                        f"got '{type(function_declarations)}'"
                    )
                    raise ValueError(msg)
                if function_declarations:
                    fds = [
                        _format_to_gapic_function_declaration(fd)
                        for fd in function_declarations
                    ]
                    gapic_tool.function_declarations.extend(fds)
            if "google_search_retrieval" in tool:
                gapic_tool.google_search_retrieval = gapic.GoogleSearchRetrieval(
                    tool["google_search_retrieval"]
                )
            if "google_search" in tool:
                gapic_tool.google_search = gapic.Tool.GoogleSearch(
                    tool["google_search"]
                )
            if "retrieval" in tool:
                gapic_tool.retrieval = gapic.Retrieval(tool["retrieval"])
            if "code_execution" in tool:
                gapic_tool.code_execution = gapic.Tool.CodeExecution(
                    tool["code_execution"]
                )
        else:
            fd = _format_to_gapic_function_declaration(tool)
            gapic_tool.function_declarations.append(fd)
    return gapic_tool


class PydanticFunctionsOutputParser(BaseOutputParser):
    """Parse an output as a pydantic object.

    This parser is used to parse the output of a chat model that uses Google Vertex
    function format to invoke functions.

    The parser extracts the function call invocation and matches them to the pydantic
    schema provided.

    An exception will be raised if the function call does not match the provided schema.

    Example:
        ```python
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
        ```
    """

    pydantic_schema: type[BaseModel] | dict[str, type[BaseModel]]

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> BaseModel:
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)
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
        msg = f"Could not parse function call: {message}"
        raise OutputParserException(msg)

    def parse(self, text: str) -> BaseModel:
        msg = "Can only parse messages"
        raise ValueError(msg)


class _FunctionCallingConfigDict(TypedDict):
    mode: gapic.FunctionCallingConfig.Mode | int
    allowed_function_names: list[str] | None


class _ToolConfigDict(TypedDict):
    function_calling_config: _FunctionCallingConfigDict


_ToolChoiceType = Union[Literal["auto", "none", "any", True], dict, list[str], str]


def _format_tool_config(tool_config: _ToolConfigDict) -> gapic.ToolConfig | None:
    if "function_calling_config" not in tool_config:
        msg = (  # type: ignore[unreachable, unused-ignore]
            "Invalid ToolConfig, missing 'function_calling_config' key. Received:\n\n"
            f"{tool_config=}"
        )
        raise ValueError(msg)
    return gapic.ToolConfig(
        function_calling_config=gapic.FunctionCallingConfig(
            **tool_config["function_calling_config"]
        )
    )


def _tool_choice_to_tool_config(
    tool_choice: _ToolChoiceType,
    all_names: list[str],
) -> GapicToolConfig | None:
    allowed_function_names: list[str] | None = None
    if tool_choice is True or tool_choice == "any":
        mode = gapic.FunctionCallingConfig.Mode.ANY
        allowed_function_names = all_names
    elif tool_choice == "auto":
        mode = gapic.FunctionCallingConfig.Mode.AUTO
    elif tool_choice == "none":
        mode = gapic.FunctionCallingConfig.Mode.NONE
    elif isinstance(tool_choice, str):
        mode = gapic.FunctionCallingConfig.Mode.ANY
        allowed_function_names = [tool_choice]
    elif isinstance(tool_choice, list):
        mode = gapic.FunctionCallingConfig.Mode.ANY
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
        elif (
            "type" in tool_choice
            and tool_choice["type"] == "function"
            and "function" in tool_choice
            and "name" in tool_choice["function"]
        ):
            mode = gapic.FunctionCallingConfig.Mode.ANY
            allowed_function_names = [tool_choice["function"]["name"]]
        else:
            msg = (  # type: ignore[unreachable, unused-ignore]
                f"Unrecognized tool choice format:\n\n{tool_choice=}\n\nShould match "
                f"VertexAI ToolConfig or FunctionCallingConfig format."
            )
            raise ValueError(msg)
    else:
        msg = f"Unrecognized tool choice format:\n\n{tool_choice=}"  # type: ignore[unreachable, unused-ignore]
        raise ValueError(msg)
    tool_config = _ToolConfigDict(
        function_calling_config=_FunctionCallingConfigDict(
            mode=mode,
            allowed_function_names=allowed_function_names,
        )
    )
    return _format_tool_config(tool_config)
