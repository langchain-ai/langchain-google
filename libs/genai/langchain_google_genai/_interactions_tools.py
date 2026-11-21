"""Tool, schema, config, and safety translation for the Gemini Interactions API."""

from __future__ import annotations

import copy
import logging
import re
import warnings
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

try:
    from google.genai._gaos.types.interactions import (
        CreateModelInteraction as _InteractionsCreateModel,
    )
    from google.genai._gaos.types.interactions import (
        GenerationConfig as _InteractionsGenerationConfig,
    )

    _HAS_NATIVE_TEMPERATURE = (
        "temperature" in _InteractionsGenerationConfig.model_fields
    )
    _HAS_NATIVE_TOP_P = "top_p" in _InteractionsGenerationConfig.model_fields
    _HAS_NATIVE_CONTINUATION_TOKEN = (
        "continuation_token" in _InteractionsCreateModel.model_fields
    )
except ImportError:  # pragma: no cover - defensive fallback if private _gaos moves
    _HAS_NATIVE_TEMPERATURE = True
    _HAS_NATIVE_TOP_P = True
    _HAS_NATIVE_CONTINUATION_TOKEN = True
from google.genai.types import (
    FunctionDeclaration,
    ThinkingConfig,
    ToolConfig,
)
from google.genai.types import Tool as GoogleTool
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as callable_as_lc_tool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import is_typeddict

from langchain_google_genai._common import SafetySettingDict
from langchain_google_genai._function_utils import (
    _ToolChoiceType,
    is_basemodel_subclass_safe,
)
from langchain_google_genai.chat_models import (
    _CUSTOM_SAMPLING_PARAMETERS,
    _INTERACTIONS_SERVER_TOOL_TYPES,
    GoogleInvalidRequestError,
    _get_default_model_profile,
    _normalize_gemini_model_name,
    _uses_fixed_sampling_and_disallows_prefill,
)

logger = logging.getLogger(__name__)

_LEGACY_SPECIAL_TOOL_KEYS: dict[str, str] = {
    "google_search": "google_search",
    "googleSearch": "google_search",
    "google_search_retrieval": "google_search",
    "googleSearchRetrieval": "google_search",
    "enterprise_web_search": "enterprise_web_search",
    "enterpriseWebSearch": "enterprise_web_search",
    "google_maps": "google_maps",
    "googleMaps": "google_maps",
    "code_execution": "code_execution",
    "codeExecution": "code_execution",
    "url_context": "url_context",
    "urlContext": "url_context",
    "computer_use": "computer_use",
    "computerUse": "computer_use",
    "file_search": "file_search",
    "fileSearch": "file_search",
    "mcp_servers": "mcp_servers",
    "mcpServers": "mcp_servers",
    "retrieval": "retrieval",
    "parallel_ai_search": "parallel_ai_search",
    "exa_ai_search": "exa_ai_search",
}

_RESPONSE_FORMAT_TYPES = frozenset({"text", "image", "audio", "video"})


def _unwrap_enum_value(value: Any) -> Any:
    """Unwrap Enum members or `tool_to_dict` serialized enum mappings."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping) and "_value_" in value:
        return value["_value_"]
    return value


def _normalize_computer_use_env(raw_env: Any) -> str | None:
    """Normalize a Computer Use `environment` enum or string to Interactions form."""
    env_val = _unwrap_enum_value(raw_env)
    if env_val is None:
        return None
    env_str = str(env_val).lower().removeprefix("environment_")
    if not env_str or env_str == "unspecified":
        return None
    return env_str


def _normalize_enum_str_list(
    raw_items: Any, *, prefixes: tuple[str, ...] = ()
) -> list[str] | None:
    """Normalize a sequence of enum/string identifiers to lowercase wire strings."""
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return None
    out: list[str] = []
    for raw in raw_items:
        unwrapped = _unwrap_enum_value(raw)
        if unwrapped is None:
            continue
        norm = str(unwrapped).lower()
        for pfx in prefixes:
            if norm.startswith(pfx):
                norm = norm.removeprefix(pfx)
                break
        if norm and norm != "unspecified":
            out.append(norm)
    return out


def _normalize_allowed_tools_entry(raw_entry: Any) -> dict[str, Any]:
    """Normalize one MCP `AllowedTools` mapping or model without mutation."""
    entry_dict: dict[str, Any]
    if isinstance(raw_entry, BaseModel):
        entry_dict = raw_entry.model_dump(exclude_none=True)
    elif isinstance(raw_entry, Mapping):
        entry_dict = {
            str(k): copy.deepcopy(v) for k, v in raw_entry.items() if v is not None
        }
    else:
        return {"tools": [str(raw_entry)]}

    if "mode" in entry_dict:
        mode_raw = _unwrap_enum_value(entry_dict.pop("mode"))
        if mode_raw is not None:
            mode_str = str(mode_raw).lower().removeprefix("mode_")
            if mode_str and mode_str != "unspecified":
                entry_dict["mode"] = mode_str

    if "tools" in entry_dict:
        tools_raw = entry_dict.pop("tools")
        if isinstance(tools_raw, str):
            entry_dict["tools"] = [tools_raw]
        elif isinstance(tools_raw, Sequence) and not isinstance(
            tools_raw, (str, bytes)
        ):
            entry_dict["tools"] = [str(t) for t in tools_raw if t is not None]

    return entry_dict


def _normalize_mcp_allowed_tools(raw_allowed: Any) -> list[dict[str, Any]] | None:
    """Normalize `mcp_server.allowed_tools` into Speakeasy `list[AllowedTools]`."""
    if raw_allowed is None:
        return None
    if isinstance(raw_allowed, str):
        return [{"tools": [raw_allowed]}]
    if isinstance(raw_allowed, (BaseModel, Mapping)):
        return [_normalize_allowed_tools_entry(raw_allowed)]
    if isinstance(raw_allowed, Sequence) and not isinstance(raw_allowed, (str, bytes)):
        if not raw_allowed:
            return []
        normalized: list[dict[str, Any]] = []
        str_bucket: list[str] = []
        for elem in raw_allowed:
            if elem is None:
                continue
            if isinstance(elem, (BaseModel, Mapping)):
                if str_bucket:
                    normalized.append({"tools": list(str_bucket)})
                    str_bucket.clear()
                normalized.append(_normalize_allowed_tools_entry(elem))
            else:
                str_bucket.append(str(elem))
        if str_bucket:
            normalized.append({"tools": list(str_bucket)})
        return normalized
    return None


def _extract_maps_lat_lng(
    tool_config: Mapping[str, Any] | ToolConfig | None,
) -> tuple[float | None, float | None]:
    """Extract `(latitude, longitude)` from `tool_config.retrieval_config.lat_lng`."""
    if tool_config is None:
        return None, None
    retrieval = (
        tool_config.get("retrieval_config") or tool_config.get("retrievalConfig")
        if isinstance(tool_config, Mapping)
        else getattr(tool_config, "retrieval_config", None)
    )
    if retrieval is None:
        return None, None
    lat_lng = (
        retrieval.get("lat_lng") or retrieval.get("latLng")
        if isinstance(retrieval, Mapping)
        else getattr(retrieval, "lat_lng", None)
    )
    if lat_lng is None:
        return None, None
    lat = (
        lat_lng.get("latitude")
        if isinstance(lat_lng, Mapping)
        else getattr(lat_lng, "latitude", None)
    )
    lng = (
        lat_lng.get("longitude")
        if isinstance(lat_lng, Mapping)
        else getattr(lat_lng, "longitude", None)
    )
    return (
        float(lat) if lat is not None else None,
        float(lng) if lng is not None else None,
    )


_GENAI_SCHEMA_KEY_MAP: dict[str, str] = {
    "any_of": "anyOf",
    "all_of": "allOf",
    "one_of": "oneOf",
    "min_items": "minItems",
    "max_items": "maxItems",
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
    "additional_properties": "additionalProperties",
    "property_ordering": "propertyOrdering",
    "prefix_items": "prefixItems",
    "defs": "$defs",
}


def _normalize_genai_schema_dump(schema: Any) -> Any:
    """Normalize a serialized `google.genai.types.Schema` dict into JSON Schema."""
    if not isinstance(schema, Mapping):
        return schema
    cleaned: dict[str, Any] = {}
    is_nullable = False
    for raw_key, raw_val in schema.items():
        if raw_val is None:
            continue
        val = _unwrap_enum_value(raw_val)
        key = _GENAI_SCHEMA_KEY_MAP.get(str(raw_key), str(raw_key))
        if key == "nullable":
            is_nullable = bool(val)
            continue
        if key in ("type", "type_") and isinstance(val, str):
            cleaned["type"] = val.lower()
        elif key in ("properties", "$defs", "definitions") and isinstance(val, Mapping):
            cleaned[key] = {
                pk: _normalize_genai_schema_dump(pv)
                for pk, pv in val.items()
                if pv is not None
            }
        elif key == "items":
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes, Mapping)):
                cleaned["items"] = [_normalize_genai_schema_dump(v) for v in val]
            else:
                cleaned["items"] = _normalize_genai_schema_dump(val)
        elif key == "additionalProperties" and isinstance(val, Mapping):
            cleaned["additionalProperties"] = _normalize_genai_schema_dump(val)
        elif (
            key in ("anyOf", "allOf", "oneOf", "prefixItems")
            and isinstance(val, Sequence)
            and not isinstance(val, (str, bytes))
        ):
            cleaned[key] = [_normalize_genai_schema_dump(v) for v in val]
        elif key == "required" and val == [] and not schema.get("properties"):
            continue
        else:
            cleaned[key] = copy.deepcopy(val)

    if (
        cleaned.get("type") == "object"
        and "properties" not in cleaned
        and not any(
            k in cleaned for k in ("additionalProperties", "anyOf", "allOf", "oneOf")
        )
    ):
        cleaned["properties"] = {}

    if is_nullable:
        if "anyOf" in cleaned and isinstance(cleaned["anyOf"], list):
            if not any(
                isinstance(opt, Mapping) and opt.get("type") == "null"
                for opt in cleaned["anyOf"]
            ):
                cleaned["anyOf"] = [*cleaned["anyOf"], {"type": "null"}]
        elif cleaned.get("type") != "null":
            meta_keys = ("description", "default", "title")
            preserved_meta = {k: cleaned.pop(k) for k in meta_keys if k in cleaned}
            base_branch = cleaned or {"type": "string"}
            cleaned = {
                "anyOf": [base_branch, {"type": "null"}],
                **preserved_meta,
            }
    return cleaned


def _convert_special_tool(
    kind: str,
    raw_cfg: Any,
    *,
    maps_lat: float | None,
    maps_lng: float | None,
) -> list[dict[str, Any]]:
    """Convert a legacy `generateContent` built-in tool config to Interactions form."""
    cfg_dict: dict[str, Any] = {}
    if isinstance(raw_cfg, BaseModel):
        cfg_dict = raw_cfg.model_dump(exclude_none=True)
    elif isinstance(raw_cfg, Mapping):
        cfg_dict = {k: v for k, v in raw_cfg.items() if v is not None}

    if kind == "google_search":
        result: dict[str, Any] = {"type": "google_search"}
        search_types = _normalize_enum_str_list(
            cfg_dict.get("search_types") or cfg_dict.get("searchTypes"),
            prefixes=("search_type_", "google_search_type_"),
        )
        if search_types:
            result["search_types"] = search_types
        return [result]

    if kind == "enterprise_web_search":
        return [{"type": "google_search", "search_types": ["enterprise_web_search"]}]

    if kind == "google_maps":
        maps_tool: dict[str, Any] = {"type": "google_maps"}
        enable_widget = cfg_dict.get(
            "enable_widget", cfg_dict.get("enableWidget", None)
        )
        if enable_widget is not None:
            maps_tool["enable_widget"] = bool(enable_widget)
        lat = cfg_dict.get("latitude", maps_lat)
        lng = cfg_dict.get("longitude", maps_lng)
        if lat is not None:
            maps_tool["latitude"] = float(lat)
        if lng is not None:
            maps_tool["longitude"] = float(lng)
        return [maps_tool]

    if kind == "code_execution":
        return [{"type": "code_execution"}]

    if kind == "url_context":
        return [{"type": "url_context"}]

    if kind == "computer_use":
        cu_tool: dict[str, Any] = {"type": "computer_use"}
        env_norm = _normalize_computer_use_env(cfg_dict.get("environment"))
        if env_norm is not None:
            cu_tool["environment"] = env_norm
        excluded = cfg_dict.get("excluded_predefined_functions") or cfg_dict.get(
            "excludedPredefinedFunctions"
        )
        if isinstance(excluded, Sequence) and not isinstance(excluded, (str, bytes)):
            if excluded:
                cu_tool["excluded_predefined_functions"] = [str(f) for f in excluded]
        disabled = _normalize_enum_str_list(
            cfg_dict.get("disabled_safety_policies")
            or cfg_dict.get("disabledSafetyPolicies"),
            prefixes=("safety_policy_", "disabled_safety_policy_"),
        )
        if disabled:
            cu_tool["disabled_safety_policies"] = disabled
        epid = cfg_dict.get(
            "enable_prompt_injection_detection",
            cfg_dict.get("enablePromptInjectionDetection"),
        )
        if epid is not None:
            cu_tool["enable_prompt_injection_detection"] = bool(epid)
        return [cu_tool]

    if kind == "file_search":
        fs_tool: dict[str, Any] = {"type": "file_search"}
        stores = cfg_dict.get("file_search_store_names") or cfg_dict.get(
            "fileSearchStoreNames"
        )
        if isinstance(stores, list):
            fs_tool["file_search_store_names"] = list(stores)
        if mf := cfg_dict.get("metadata_filter") or cfg_dict.get("metadataFilter"):
            fs_tool["metadata_filter"] = str(mf)
        if (tk := cfg_dict.get("top_k") or cfg_dict.get("topK")) is not None:
            fs_tool["top_k"] = int(tk)
        return [fs_tool]

    if kind == "mcp_servers":
        servers = raw_cfg if isinstance(raw_cfg, Sequence) else []
        out_servers: list[dict[str, Any]] = []
        for srv in servers:
            srv_dict = (
                srv.model_dump(exclude_none=True)
                if isinstance(srv, BaseModel)
                else dict(srv)
                if isinstance(srv, Mapping)
                else {}
            )
            mcp_item: dict[str, Any] = {"type": "mcp_server"}
            if srv_dict.get("name"):
                mcp_item["name"] = str(srv_dict["name"])
            url = srv_dict.get("url") or srv_dict.get("streamable_http_url")
            if not url and isinstance(srv_dict.get("streamable_http_transport"), dict):
                url = srv_dict["streamable_http_transport"].get("url")
            if url:
                mcp_item["url"] = str(url)
            headers = srv_dict.get("headers")
            if not headers and isinstance(
                srv_dict.get("streamable_http_transport"), dict
            ):
                headers = srv_dict["streamable_http_transport"].get("headers")
            if isinstance(headers, Mapping):
                mcp_item["headers"] = dict(headers)
            raw_allowed = (
                srv_dict["allowed_tools"]
                if "allowed_tools" in srv_dict
                else srv_dict.get("allowedTools")
            )
            norm_allowed = _normalize_mcp_allowed_tools(raw_allowed)
            if norm_allowed is not None:
                mcp_item["allowed_tools"] = norm_allowed
            out_servers.append(mcp_item)
        return out_servers

    if kind == "retrieval":
        ret: dict[str, Any] = {"type": "retrieval", **copy.deepcopy(cfg_dict)}
        if "retrievalTypes" in ret and "retrieval_types" not in ret:
            ret["retrieval_types"] = ret.pop("retrievalTypes")
        else:
            ret.pop("retrievalTypes", None)
        if "retrieval_types" in ret:
            norm_rt = _normalize_enum_str_list(
                ret.pop("retrieval_types"), prefixes=("retrieval_type_",)
            )
            if norm_rt:
                ret["retrieval_types"] = norm_rt
        return [ret]
    if kind == "parallel_ai_search":
        return [
            {
                "type": "retrieval",
                "retrieval_types": ["parallel_ai_search"],
                "parallel_ai_search_config": copy.deepcopy(cfg_dict),
            }
        ]
    if kind == "exa_ai_search":
        return [
            {
                "type": "retrieval",
                "retrieval_types": ["exa_ai_search"],
                "exa_ai_search_config": copy.deepcopy(cfg_dict),
            }
        ]
    return []


def _function_dict_from_openai(oai_tool: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI `{"type": "function", "function": {...}}` dict cleanly."""
    fn = oai_tool["function"]
    out: dict[str, Any] = {"type": "function", "name": str(fn["name"])}
    if desc := fn.get("description"):
        out["description"] = str(desc)
    params = fn.get("parameters")
    if isinstance(params, Mapping) and params:
        copied_params = copy.deepcopy(dict(params))
        copied_params.pop("strict", None)
        if "type" not in copied_params and "properties" in copied_params:
            copied_params["type"] = "object"
        out["parameters"] = copied_params
    else:
        out["parameters"] = {"type": "object", "properties": {}}
    return out


def _convert_single_function_tool(tool_item: Any) -> dict[str, Any]:
    """Convert a single callable/tool/schema/dict into an Interactions `function`."""
    if isinstance(tool_item, FunctionDeclaration):
        dumped = tool_item.model_dump(exclude_none=True)
        out: dict[str, Any] = {"type": "function", "name": str(dumped["name"])}
        if dumped.get("description"):
            out["description"] = str(dumped["description"])
        raw_params = dumped.get("parameters_json_schema") or dumped.get("parameters")
        if raw_params:
            out["parameters"] = _normalize_genai_schema_dump(raw_params)
        else:
            out["parameters"] = {"type": "object", "properties": {}}
        return out

    if isinstance(tool_item, Runnable) and not isinstance(tool_item, BaseTool):
        return _function_dict_from_openai(convert_to_openai_tool(tool_item.as_tool()))

    if isinstance(tool_item, type) and issubclass(tool_item, BaseModelV1):
        schema_v1 = copy.deepcopy(tool_item.schema())
        name_v1 = schema_v1.pop("title", tool_item.__name__)
        desc_v1 = schema_v1.pop("description", tool_item.__doc__ or "")
        out_v1: dict[str, Any] = {"type": "function", "name": str(name_v1)}
        if desc_v1:
            out_v1["description"] = str(desc_v1)
        schema_v1.setdefault("type", "object")
        schema_v1.setdefault("properties", {})
        out_v1["parameters"] = schema_v1
        return out_v1

    if isinstance(tool_item, BaseTool):
        if isinstance(tool_item.args_schema, type) and issubclass(
            tool_item.args_schema, BaseModelV1
        ):
            params_v1 = copy.deepcopy(tool_item.args_schema.schema())
            params_v1.pop("title", None)
            params_v1.setdefault("type", "object")
            params_v1.setdefault("properties", {})
            res_bt: dict[str, Any] = {"type": "function", "name": tool_item.name}
            if tool_item.description:
                res_bt["description"] = tool_item.description
            res_bt["parameters"] = params_v1
            return res_bt
        return _function_dict_from_openai(convert_to_openai_tool(tool_item))

    if isinstance(tool_item, Mapping):
        # OpenAI tool dict
        if tool_item.get("type") == "function" and isinstance(
            tool_item.get("function"), Mapping
        ):
            return _function_dict_from_openai(tool_item)
        # Already an Interactions function dict
        if tool_item.get("type") == "function" and "name" in tool_item:
            copied = copy.deepcopy(dict(tool_item))
            copied.pop("cache_control", None)
            copied.pop("strict", None)
            params = copied.get("parameters")
            if isinstance(params, Mapping):
                copied["parameters"] = _normalize_genai_schema_dump(params)
            elif params is None:
                copied["parameters"] = {"type": "object", "properties": {}}
            return copied
        # Anthropic tool dict with input_schema
        if "name" in tool_item and "input_schema" in tool_item:
            anth: dict[str, Any] = {"type": "function", "name": str(tool_item["name"])}
            if tool_item.get("description"):
                anth["description"] = str(tool_item["description"])
            schema_in = copy.deepcopy(dict(tool_item["input_schema"]))
            schema_in.setdefault("type", "object")
            schema_in.setdefault("properties", {})
            anth["parameters"] = schema_in
            return anth
        # FunctionDeclaration-like dict {"name": ..., "parameters": ...}
        if "name" in tool_item and "properties" not in tool_item:
            fd_out: dict[str, Any] = {
                "type": "function",
                "name": str(tool_item["name"]),
            }
            if tool_item.get("description"):
                fd_out["description"] = str(tool_item["description"])
            raw_p = tool_item.get("parameters_json_schema") or tool_item.get(
                "parameters"
            )
            if isinstance(raw_p, Mapping) and raw_p:
                fd_out["parameters"] = _normalize_genai_schema_dump(raw_p)
            else:
                fd_out["parameters"] = {"type": "object", "properties": {}}
            return fd_out
        # Raw JSON Schema dict (with title/properties)
        return _function_dict_from_openai(
            convert_to_openai_tool(copy.deepcopy(dict(tool_item)))
        )

    if (
        isinstance(tool_item, type) and is_basemodel_subclass_safe(tool_item)
    ) or is_typeddict(tool_item):
        return _function_dict_from_openai(convert_to_openai_tool(tool_item))

    if callable(tool_item):
        return _function_dict_from_openai(
            convert_to_openai_tool(callable_as_lc_tool()(tool_item))
        )

    return _function_dict_from_openai(convert_to_openai_tool(tool_item))


def _normalize_server_tool_dict(
    item: Mapping[str, Any],
    *,
    maps_lat: float | None,
    maps_lng: float | None,
) -> dict[str, Any]:
    """Normalize an Interactions server tool dict cleanly without mutating `item`."""
    copied_srv: dict[str, Any] = {}
    for k, v in item.items():
        if isinstance(v, BaseModel):
            copied_srv[str(k)] = v.model_dump(exclude_none=True)
        else:
            copied_srv[str(k)] = copy.deepcopy(v)

    item_type = copied_srv.get("type")

    if item_type == "google_maps":
        if "enableWidget" in copied_srv and "enable_widget" not in copied_srv:
            copied_srv["enable_widget"] = copied_srv.pop("enableWidget")
        else:
            copied_srv.pop("enableWidget", None)
        if copied_srv.get("enable_widget") is not None:
            copied_srv["enable_widget"] = bool(copied_srv["enable_widget"])
        else:
            copied_srv.pop("enable_widget", None)
        if copied_srv.get("latitude") is None:
            copied_srv.pop("latitude", None)
            if maps_lat is not None:
                copied_srv["latitude"] = float(maps_lat)
        else:
            copied_srv["latitude"] = float(copied_srv["latitude"])
        if copied_srv.get("longitude") is None:
            copied_srv.pop("longitude", None)
            if maps_lng is not None:
                copied_srv["longitude"] = float(maps_lng)
        else:
            copied_srv["longitude"] = float(copied_srv["longitude"])

    elif item_type == "mcp_server":
        has_allowed = "allowed_tools" in item or "allowedTools" in item
        raw_allowed = (
            item["allowed_tools"]
            if "allowed_tools" in item
            else item.get("allowedTools")
        )
        copied_srv.pop("allowed_tools", None)
        copied_srv.pop("allowedTools", None)
        if has_allowed:
            norm_allowed = _normalize_mcp_allowed_tools(raw_allowed)
            if norm_allowed is not None:
                copied_srv["allowed_tools"] = norm_allowed
        if copied_srv.get("headers") is None:
            copied_srv.pop("headers", None)

    elif item_type == "computer_use":
        if "environment" in copied_srv:
            env_norm = _normalize_computer_use_env(copied_srv.pop("environment"))
            if env_norm is not None:
                copied_srv["environment"] = env_norm
        raw_policies = (
            copied_srv.pop("disabled_safety_policies")
            if "disabled_safety_policies" in copied_srv
            else copied_srv.pop("disabledSafetyPolicies", None)
        )
        copied_srv.pop("disabledSafetyPolicies", None)
        norm_policies = _normalize_enum_str_list(
            raw_policies,
            prefixes=("safety_policy_", "disabled_safety_policy_"),
        )
        if norm_policies:
            copied_srv["disabled_safety_policies"] = norm_policies
        raw_excl = (
            copied_srv.pop("excluded_predefined_functions")
            if "excluded_predefined_functions" in copied_srv
            else copied_srv.pop("excludedPredefinedFunctions", None)
        )
        copied_srv.pop("excludedPredefinedFunctions", None)
        if isinstance(raw_excl, Sequence) and not isinstance(raw_excl, (str, bytes)):
            if raw_excl:
                copied_srv["excluded_predefined_functions"] = [
                    str(fn) for fn in raw_excl
                ]
        if "enablePromptInjectionDetection" in copied_srv:
            epid = copied_srv.pop("enablePromptInjectionDetection")
            if "enable_prompt_injection_detection" not in copied_srv and (
                epid is not None
            ):
                copied_srv["enable_prompt_injection_detection"] = bool(epid)
        if copied_srv.get("enable_prompt_injection_detection") is not None:
            copied_srv["enable_prompt_injection_detection"] = bool(
                copied_srv["enable_prompt_injection_detection"]
            )
        else:
            copied_srv.pop("enable_prompt_injection_detection", None)

    elif item_type == "google_search":
        raw_st = (
            copied_srv.pop("search_types")
            if "search_types" in copied_srv
            else copied_srv.pop("searchTypes", None)
        )
        copied_srv.pop("searchTypes", None)
        norm_st = _normalize_enum_str_list(
            raw_st, prefixes=("search_type_", "google_search_type_")
        )
        if norm_st:
            copied_srv["search_types"] = norm_st

    elif item_type == "retrieval":
        raw_rt = (
            copied_srv.pop("retrieval_types")
            if "retrieval_types" in copied_srv
            else copied_srv.pop("retrievalTypes", None)
        )
        copied_srv.pop("retrievalTypes", None)
        norm_rt = _normalize_enum_str_list(raw_rt, prefixes=("retrieval_type_",))
        if norm_rt:
            copied_srv["retrieval_types"] = norm_rt

    elif item_type == "file_search":
        for camel, snake in (
            ("fileSearchStoreNames", "file_search_store_names"),
            ("metadataFilter", "metadata_filter"),
            ("topK", "top_k"),
        ):
            if camel in copied_srv:
                cval = copied_srv.pop(camel)
                if snake not in copied_srv and cval is not None:
                    copied_srv[snake] = cval

    return copied_srv


def _format_interactions_tools(
    tools: Sequence[Any] | None = None,
    functions: Sequence[Any] | None = None,
    *,
    tool_config: Mapping[str, Any] | ToolConfig | None = None,
) -> list[dict[str, Any]] | None:
    """Translate LangChain and GenAI tools into Interactions API `ToolParam` dicts."""
    source_items: list[Any] = []
    if tools:
        source_items.extend(tools)
    if functions:
        source_items.extend(functions)
    if not source_items:
        return None

    maps_lat, maps_lng = _extract_maps_lat_lng(tool_config)
    formatted: list[dict[str, Any]] = []

    for item in source_items:
        if isinstance(item, GoogleTool):
            dumped = item.model_dump(exclude_none=True)
            for key, val in dumped.items():
                if key == "function_declarations" and isinstance(val, list):
                    formatted.extend(
                        _convert_single_function_tool(fd)
                        for fd in (item.function_declarations or [])
                    )
                elif (mapped := _LEGACY_SPECIAL_TOOL_KEYS.get(key)) is not None:
                    formatted.extend(
                        _convert_special_tool(
                            mapped, val, maps_lat=maps_lat, maps_lng=maps_lng
                        )
                    )
            continue

        if isinstance(item, Mapping):
            item_type = item.get("type")
            if item_type in _INTERACTIONS_SERVER_TOOL_TYPES:
                formatted.append(
                    _normalize_server_tool_dict(
                        item, maps_lat=maps_lat, maps_lng=maps_lng
                    )
                )
                continue

            special_hits = [
                (k, _LEGACY_SPECIAL_TOOL_KEYS[k])
                for k in item
                if k in _LEGACY_SPECIAL_TOOL_KEYS and item.get(k) is not None
            ]
            if len(special_hits) > 1:
                found = [k for k, _ in special_hits]
                msg = (
                    "A single tool dict cannot have multiple special tool types. "
                    f"Found: {found}."
                )
                raise GoogleInvalidRequestError(msg)
            if len(special_hits) == 1:
                orig_key, mapped_kind = special_hits[0]
                formatted.extend(
                    _convert_special_tool(
                        mapped_kind,
                        item[orig_key],
                        maps_lat=maps_lat,
                        maps_lng=maps_lng,
                    )
                )
                continue

            if item.get("function_declarations"):
                fds = item["function_declarations"]
                if not isinstance(fds, Sequence) or isinstance(fds, (str, bytes)):
                    msg = f"function_declarations should be a list, got {type(fds)}"
                    raise GoogleInvalidRequestError(msg)
                formatted.extend(_convert_single_function_tool(fd) for fd in fds)
                continue

        formatted.append(_convert_single_function_tool(item))

    return formatted or None


def _format_interactions_tool_choice(
    tool_choice: _ToolChoiceType | bool | None,
    tool_config: Mapping[str, Any] | ToolConfig | None,
    formatted_tools: Sequence[dict[str, Any]] | None,
) -> str | dict[str, Any] | None:
    """Convert `tool_choice` / `tool_config` into Interactions `tool_choice`."""
    fcc: Any = None
    if tool_config is not None:
        fcc = (
            tool_config.get("function_calling_config")
            or tool_config.get("functionCallingConfig")
            if isinstance(tool_config, Mapping)
            else getattr(tool_config, "function_calling_config", None)
        )

    has_explicit_choice = tool_choice is not None and tool_choice is not False
    if has_explicit_choice and fcc is not None:
        msg = "Cannot specify both tool_choice and tool_config.function_calling_config."
        raise ValueError(msg)

    if has_explicit_choice and not formatted_tools:
        msg = (
            f"Received {tool_choice=} but no tools were provided. "
            "'tool_choice' can only be specified if 'tools' is specified."
        )
        raise ValueError(msg)

    fn_names = [
        str(t["name"])
        for t in (formatted_tools or [])
        if t.get("type") == "function" and t.get("name")
    ]
    # Built-in server tools only: omit function tool_choice
    if not fn_names:
        return None

    if not has_explicit_choice and fcc is not None:
        mode_raw = (
            fcc.get("mode") if isinstance(fcc, Mapping) else getattr(fcc, "mode", None)
        )
        allowed = (
            fcc.get("allowed_function_names") or fcc.get("allowedFunctionNames")
            if isinstance(fcc, Mapping)
            else getattr(fcc, "allowed_function_names", None)
        )
        mode_str = str(_unwrap_enum_value(mode_raw or "auto")).lower()
        if allowed:
            return {"allowed_tools": {"mode": mode_str, "tools": list(allowed)}}
        return mode_str

    if not has_explicit_choice:
        return None

    if tool_choice is True or tool_choice in ("any", "ANY", "required", "REQUIRED"):
        return "any"
    if tool_choice in ("auto", "AUTO"):
        return "auto"
    if tool_choice in ("none", "NONE"):
        return "none"
    if tool_choice in ("validated", "VALIDATED"):
        return "validated"
    if isinstance(tool_choice, str):
        return {"allowed_tools": {"mode": "any", "tools": [tool_choice]}}
    if isinstance(tool_choice, list):
        return {"allowed_tools": {"mode": "any", "tools": list(tool_choice)}}
    if isinstance(tool_choice, Mapping):
        if "allowed_tools" in tool_choice and isinstance(
            tool_choice["allowed_tools"], Mapping
        ):
            at = dict(tool_choice["allowed_tools"])
            if "mode" in at and at["mode"] is not None:
                at["mode"] = str(_unwrap_enum_value(at["mode"])).lower()
            return {"allowed_tools": copy.deepcopy(at)}
        if tool_choice.get("type") == "function" and isinstance(
            tool_choice.get("function"), Mapping
        ):
            name = tool_choice["function"].get("name")
            if name:
                return {"allowed_tools": {"mode": "any", "tools": [str(name)]}}
        if tool_choice.get("type") == "tool" and tool_choice.get("name"):
            return {
                "allowed_tools": {"mode": "any", "tools": [str(tool_choice["name"])]}
            }
        if tool_choice.get("type") in ("any", "auto", "none", "validated"):
            return str(tool_choice["type"])
        inner = (
            tool_choice["function_calling_config"]
            if "function_calling_config" in tool_choice
            else tool_choice
        )
        if isinstance(inner, Mapping) and "mode" in inner:
            mode = str(_unwrap_enum_value(inner["mode"])).lower()
            allowed_names = inner.get("allowed_function_names")
            if allowed_names:
                return {"allowed_tools": {"mode": mode, "tools": list(allowed_names)}}
            return mode
    msg = f"Unrecognized tool_choice format: {tool_choice!r}"
    raise GoogleInvalidRequestError(msg)


def _model_supports_thinking_level(
    model_name: str, profile: Mapping[str, Any] | None, level: str
) -> bool:
    """Return True only when the model profile explicitly lists `level`."""
    if profile is not None:
        levels = profile.get("reasoning_effort_levels")
    else:
        model_id = re.sub(r"-\d{3}$", "", _normalize_gemini_model_name(model_name))
        levels = _get_default_model_profile(model_id).get("reasoning_effort_levels")
    return isinstance(levels, Sequence) and level in levels


def _model_supports_minimal_thinking(
    model_name: str, profile: Mapping[str, Any] | None
) -> bool:
    """Return True only when the model profile explicitly lists `'minimal'`."""
    return _model_supports_thinking_level(model_name, profile, "minimal")


def _model_supports_xhigh_thinking(
    model_name: str, profile: Mapping[str, Any] | None
) -> bool:
    """Return True only when the model profile explicitly lists `'xhigh'`."""
    return _model_supports_thinking_level(model_name, profile, "xhigh")


def _resolve_interactions_thinking(
    *,
    effective_model: str,
    profile: Mapping[str, Any] | None,
    raw_thinking_config: Mapping[str, Any] | ThinkingConfig | None,
    thinking_level: str | None,
    thinking_budget: int | None,
    include_thoughts: bool | None,
    thinking_summaries: str | None,
) -> tuple[str | None, str | None]:
    """Resolve `(thinking_level, thinking_summaries)` for Interactions config."""
    merged: dict[str, Any] = {}
    if raw_thinking_config is not None:
        if isinstance(raw_thinking_config, ThinkingConfig):
            merged.update(raw_thinking_config.model_dump(exclude_none=True))
        elif isinstance(raw_thinking_config, Mapping):
            raw_dict = dict(raw_thinking_config)
            extra_summaries = raw_dict.pop("thinking_summaries", None)
            if raw_dict:
                merged.update(
                    ThinkingConfig.model_validate(raw_dict).model_dump(
                        exclude_none=True
                    )
                )
            if extra_summaries is not None:
                merged["thinking_summaries"] = extra_summaries

    if thinking_level is not None:
        merged["thinking_level"] = thinking_level
    if thinking_budget is not None:
        merged["thinking_budget"] = thinking_budget
    if include_thoughts is not None:
        merged["include_thoughts"] = include_thoughts
    if thinking_summaries is not None:
        merged["thinking_summaries"] = thinking_summaries

    if not merged:
        return None, None

    if "thinking_level" in merged and "thinking_budget" in merged:
        warnings.warn(
            "Both 'thinking_level' and 'thinking_budget' were set after merging "
            "thinking configuration values. 'thinking_level' takes precedence "
            "for Gemini 3+ models; 'thinking_budget' will be ignored.",
            UserWarning,
            stacklevel=3,
        )
        merged.pop("thinking_budget")

    supports_minimal = _model_supports_minimal_thinking(effective_model, profile)
    supports_xhigh = _model_supports_xhigh_thinking(effective_model, profile)
    resolved_level: str | None = None
    resolved_summaries: str | None = None

    if "thinking_level" in merged and merged["thinking_level"] is not None:
        lvl_raw = str(_unwrap_enum_value(merged["thinking_level"])).lower()
        lvl_raw = lvl_raw.removeprefix("thinking_level_")
        if lvl_raw == "minimal" and not supports_minimal:
            resolved_level = "low"
        elif lvl_raw == "xhigh" and not supports_xhigh:
            # Models without `xhigh` reject it, so fall back to the closest level.
            warnings.warn(
                f"Model '{effective_model}' does not list 'xhigh' in its profile's "
                "reasoning_effort_levels; using 'high' instead. Set a profile with "
                "'xhigh' in reasoning_effort_levels to send it as-is.",
                UserWarning,
                stacklevel=3,
            )
            resolved_level = "high"
        else:
            resolved_level = lvl_raw
    elif "thinking_budget" in merged and merged["thinking_budget"] is not None:
        budget = int(merged["thinking_budget"])
        if budget < -1:
            msg = f"thinking_budget must be >= -1, got {budget}"
            raise GoogleInvalidRequestError(msg)
        if budget == -1:
            resolved_level = None
        elif budget == 0:
            resolved_level = "minimal" if supports_minimal else "low"
            resolved_summaries = "none"
        elif budget <= 1024:
            resolved_level = "low"
        elif budget <= 8192:
            resolved_level = "medium"
        else:
            resolved_level = "high"

    if "include_thoughts" in merged and merged["include_thoughts"] is not None:
        resolved_summaries = "auto" if bool(merged["include_thoughts"]) else "none"
    if "thinking_summaries" in merged and merged["thinking_summaries"] is not None:
        resolved_summaries = str(
            _unwrap_enum_value(merged["thinking_summaries"])
        ).lower()

    return resolved_level, resolved_summaries


def _format_interactions_speech_config(
    raw_speech: Any, *, voice_name: str | None = None, is_tts_model: bool = False
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Convert GenAI or Interactions speech config to `SpeechConfigUnionParam`."""
    if raw_speech is None:
        if voice_name:
            return [{"voice": voice_name}]
        if is_tts_model:
            return [{"voice": "Kore"}]
        return None

    if isinstance(raw_speech, list):
        return [copy.deepcopy(dict(s)) for s in raw_speech if isinstance(s, Mapping)]

    speech_dict = (
        raw_speech.model_dump(exclude_none=True)
        if isinstance(raw_speech, BaseModel)
        else dict(raw_speech)
        if isinstance(raw_speech, Mapping)
        else {}
    )
    if "speakers" in speech_dict:
        return copy.deepcopy(speech_dict)
    if "voice" in speech_dict:
        return [copy.deepcopy(speech_dict)]

    multi = speech_dict.get("multi_speaker_voice_config") or speech_dict.get(
        "multiSpeakerVoiceConfig"
    )
    if isinstance(multi, Mapping):
        cfgs = multi.get("speaker_voice_configs") or multi.get(
            "speakerVoiceConfigs", []
        )
        speakers: list[dict[str, Any]] = []
        for item in cfgs:
            if not isinstance(item, Mapping):
                continue
            vc = item.get("voice_config") or item.get("voiceConfig") or {}
            pvc = vc.get("prebuilt_voice_config") or vc.get("prebuiltVoiceConfig") or {}
            vname = pvc.get("voice_name") or pvc.get("voiceName")
            sp: dict[str, Any] = {}
            if item.get("speaker"):
                sp["speaker"] = str(item["speaker"])
            if vname:
                sp["voice"] = str(vname)
            if speech_dict.get("language_code"):
                sp["language"] = str(speech_dict["language_code"])
            if sp:
                speakers.append(sp)
        if speakers:
            return {"speakers": speakers}

    vc = speech_dict.get("voice_config") or speech_dict.get("voiceConfig")
    if isinstance(vc, Mapping):
        pvc = vc.get("prebuilt_voice_config") or vc.get("prebuiltVoiceConfig") or {}
        vname = pvc.get("voice_name") or pvc.get("voiceName")
        entry: dict[str, Any] = {}
        if vname:
            entry["voice"] = str(vname)
        if speech_dict.get("language_code"):
            entry["language"] = str(speech_dict["language_code"])
        if entry:
            return [entry]
    return None


def _format_interactions_response_format(
    *,
    effective_model: str,
    response_format_kwarg: Any,
    response_mime_type: str | None,
    response_schema: Any,
    response_json_schema: Any,
    response_modalities: Sequence[Any] | None,
    image_config: Mapping[str, Any] | None,
    pydantic_schema_hint: type | None = None,
) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, type | None]:
    """Build `response_format` and return `(format_param, pydantic_cls)`."""
    pydantic_cls: type | None = (
        pydantic_schema_hint
        if isinstance(pydantic_schema_hint, type)
        and issubclass(pydantic_schema_hint, (BaseModel, BaseModelV1))
        else None
    )

    def _schema_from_any(candidate: Any) -> dict[str, Any] | None:
        nonlocal pydantic_cls
        if candidate is None:
            return None
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            pydantic_cls = candidate
            return copy.deepcopy(candidate.model_json_schema())
        if isinstance(candidate, type) and issubclass(candidate, BaseModelV1):
            pydantic_cls = candidate
            return copy.deepcopy(candidate.schema())
        if is_typeddict(candidate):
            return copy.deepcopy(convert_to_json_schema(candidate))
        if isinstance(candidate, Mapping):
            return copy.deepcopy(dict(candidate))
        return None

    explicit_mime = response_mime_type
    explicit_schema = (
        _schema_from_any(response_json_schema)
        if response_json_schema is not None
        else _schema_from_any(response_schema)
    )

    # Direct Interactions-style response_format list or dict
    if isinstance(response_format_kwarg, list):
        return [copy.deepcopy(dict(x)) for x in response_format_kwarg], None

    if isinstance(response_format_kwarg, type) and issubclass(
        response_format_kwarg, (BaseModel, BaseModelV1)
    ):
        rf_schema = _schema_from_any(response_format_kwarg)
        if explicit_mime is None:
            explicit_mime = "application/json"
        if explicit_schema is None:
            explicit_schema = rf_schema
    elif isinstance(response_format_kwarg, Mapping):
        rf_type = response_format_kwarg.get("type")
        if rf_type in _RESPONSE_FORMAT_TYPES:
            return copy.deepcopy(dict(response_format_kwarg)), pydantic_cls
        if rf_type == "json_object":
            if explicit_mime is None:
                explicit_mime = "application/json"
        elif rf_type == "json_schema":
            if explicit_mime is None:
                explicit_mime = "application/json"
            js_inner = response_format_kwarg.get("json_schema")
            if explicit_schema is None and isinstance(js_inner, Mapping):
                explicit_schema = _schema_from_any(js_inner.get("schema"))

    if explicit_mime == "text/x.enum":
        msg = (
            "response_mime_type='text/x.enum' is not supported by Interactions. "
            "Define enums inside a JSON schema with mime_type='application/json'."
        )
        raise GoogleInvalidRequestError(msg)

    if explicit_schema is not None:
        if explicit_mime is not None and explicit_mime != "application/json":
            msg = (
                "JSON schema structured output is only supported when "
                "response_mime_type is 'application/json'."
            )
            raise GoogleInvalidRequestError(msg)
        return (
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": explicit_schema,
            },
            pydantic_cls,
        )

    if explicit_mime == "application/json":
        return {"type": "text", "mime_type": "application/json"}, pydantic_cls
    if explicit_mime == "text/plain":
        return {"type": "text", "mime_type": "text/plain"}, pydantic_cls

    modalities_upper = {
        str(_unwrap_enum_value(m)).upper().removeprefix("MODALITY_")
        for m in (response_modalities or [])
    }
    norm_model = _normalize_gemini_model_name(effective_model)
    is_tts = norm_model.endswith("-tts") or "-tts-" in norm_model
    is_img_model = (
        norm_model.endswith("-image")
        or "-image-" in norm_model
        or "nano-banana" in norm_model
    )

    formats: list[dict[str, Any]] = []
    if "TEXT" in modalities_upper:
        formats.append({"type": "text"})

    if (
        "IMAGE" in modalities_upper
        or image_config is not None
        or (is_img_model and not modalities_upper)
    ):
        img_fmt: dict[str, Any] = {"type": "image"}
        if isinstance(image_config, Mapping):
            if image_config.get("aspect_ratio") or image_config.get("aspectRatio"):
                img_fmt["aspect_ratio"] = str(
                    image_config.get("aspect_ratio") or image_config.get("aspectRatio")
                )
            if image_config.get("image_size") or image_config.get("imageSize"):
                img_fmt["image_size"] = str(
                    image_config.get("image_size") or image_config.get("imageSize")
                )
            out_mt = (
                image_config.get("mime_type")
                or image_config.get("output_mime_type")
                or image_config.get("outputMimeType")
            )
            if out_mt:
                img_fmt["mime_type"] = str(out_mt)
            if image_config.get("delivery"):
                img_fmt["delivery"] = str(image_config["delivery"])
        if "IMAGE" in modalities_upper or image_config is not None:
            formats.append(img_fmt)
        elif is_img_model:
            formats.extend([{"type": "text"}, img_fmt])

    if "AUDIO" in modalities_upper or (is_tts and "AUDIO" not in modalities_upper):
        formats.append({"type": "audio"})

    if not formats:
        return None, pydantic_cls
    if len(formats) == 1:
        return formats[0], pydantic_cls
    return formats, pydantic_cls


def _format_interactions_safety_settings(
    safety_settings: SafetySettingDict | Sequence[Any] | None,
) -> list[dict[str, str]] | None:
    """Map safety settings onto Interactions lowercase `{type, threshold}` dicts."""
    if not safety_settings:
        return None

    def _norm_cat(raw_cat: Any) -> str:
        val = str(_unwrap_enum_value(raw_cat)).lower()
        return val.removeprefix("harm_category_")

    def _norm_thresh(raw_thresh: Any) -> str:
        val = str(_unwrap_enum_value(raw_thresh)).lower()
        return val.removeprefix("harm_block_threshold_")

    result: list[dict[str, str]] = []
    if isinstance(safety_settings, Mapping):
        for cat, thresh in safety_settings.items():
            c_str = _norm_cat(cat)
            t_str = _norm_thresh(thresh)
            if c_str != "unspecified" and t_str != "unspecified":
                result.append({"type": c_str, "threshold": t_str})
        return result or None

    if isinstance(safety_settings, Sequence) and not isinstance(
        safety_settings, (str, bytes)
    ):
        for item in safety_settings:
            s_dict = (
                item.model_dump(exclude_none=True)
                if isinstance(item, BaseModel)
                else dict(item)
                if isinstance(item, Mapping)
                else None
            )
            if s_dict is None:
                msg = f"Invalid safety_settings item: {item!r}"
                raise TypeError(msg)
            raw_c = s_dict.get("type") or s_dict.get("category")
            raw_t = s_dict.get("threshold")
            if raw_c is None or raw_t is None:
                continue
            entry: dict[str, str] = {
                "type": _norm_cat(raw_c),
                "threshold": _norm_thresh(raw_t),
            }
            if s_dict.get("method") is not None:
                m_str = str(_unwrap_enum_value(s_dict["method"])).lower()
                m_str = m_str.removeprefix("harm_block_method_")
                if m_str != "unspecified":
                    entry["method"] = m_str
            result.append(entry)
        return result or None

    msg = "safety_settings must be a dict[HarmCategory, HarmBlockThreshold] or list"
    raise TypeError(msg)


def _format_interactions_agent_config(
    agent: str, raw_agent_config: Mapping[str, Any] | BaseModel | None
) -> dict[str, Any] | None:
    """Normalize `agent_config`, inferring discriminator `type` when omitted."""
    if raw_agent_config is None:
        return None
    cfg = (
        raw_agent_config.model_dump(exclude_none=True)
        if isinstance(raw_agent_config, BaseModel)
        else copy.deepcopy(dict(raw_agent_config))
    )
    if "type" not in cfg:
        norm = agent.lower()
        if norm.startswith("deep-research"):
            cfg["type"] = "deep-research"
        elif norm.startswith("antigravity"):
            cfg["type"] = "antigravity"
        elif norm.startswith("code-mender"):
            cfg["type"] = "code-mender"
        else:
            cfg["type"] = "dynamic"
    return cfg


def _build_interactions_generation_config(
    *,
    effective_model: str,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    frequency_penalty: float | None,
    presence_penalty: float | None,
    max_output_tokens: int | None,
    seed: int | None,
    candidate_count: int | None,
    stop_sequences: list[str] | None,
    thinking_level: str | None,
    thinking_summaries: str | None,
    tool_choice_param: str | dict[str, Any] | None,
    speech_config: list[dict[str, Any]] | dict[str, Any] | None,
    transcription_config: Mapping[str, Any] | None = None,
    video_config: Mapping[str, Any] | None = None,
    logprobs: Any = None,
    response_logprobs: Any = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build `(generation_config, extra_gen_config)` with SDK 2.25.0 compatibility."""
    sampling: dict[str, Any] = {}
    if temperature is not None:
        sampling["temperature"] = float(temperature)
    if top_p is not None:
        sampling["top_p"] = float(top_p)
    if top_k is not None:
        sampling["top_k"] = int(top_k)

    if _uses_fixed_sampling_and_disallows_prefill(effective_model):
        ignored = [p for p in _CUSTOM_SAMPLING_PARAMETERS if p in sampling]
        if ignored:
            warnings.warn(
                f"Model '{effective_model}' uses fixed sampling defaults; the "
                f"sampling parameter(s) {', '.join(ignored)} will be ignored.",
                UserWarning,
                stacklevel=3,
            )
            for p in ignored:
                sampling.pop(p, None)

    unsupported_present: list[str] = []
    if "top_k" in sampling:
        sampling.pop("top_k", None)
        unsupported_present.append("top_k")
    if frequency_penalty is not None:
        unsupported_present.append("frequency_penalty")
    if presence_penalty is not None:
        unsupported_present.append("presence_penalty")
    if candidate_count is not None and candidate_count > 1:
        unsupported_present.append("n/candidate_count > 1")
    if logprobs is not None or response_logprobs:
        unsupported_present.append("logprobs")
    if unsupported_present:
        warnings.warn(
            "Gemini Interactions API does not support "
            f"{', '.join(unsupported_present)}; omitting from request.",
            UserWarning,
            stacklevel=3,
        )

    gen_cfg: dict[str, Any] = {}
    if max_output_tokens is not None:
        gen_cfg["max_output_tokens"] = int(max_output_tokens)
    if seed is not None:
        gen_cfg["seed"] = int(seed)
    if stop_sequences:
        gen_cfg["stop_sequences"] = list(stop_sequences)
    if thinking_level is not None:
        gen_cfg["thinking_level"] = thinking_level
    if thinking_summaries is not None:
        gen_cfg["thinking_summaries"] = thinking_summaries
    if tool_choice_param is not None:
        gen_cfg["tool_choice"] = tool_choice_param
    if speech_config is not None:
        gen_cfg["speech_config"] = speech_config
    if transcription_config is not None:
        gen_cfg["transcription_config"] = copy.deepcopy(dict(transcription_config))
    if video_config is not None:
        gen_cfg["video_config"] = copy.deepcopy(dict(video_config))

    extra_gen_cfg: dict[str, Any] = {}
    if "temperature" in sampling:
        if _HAS_NATIVE_TEMPERATURE:
            gen_cfg["temperature"] = sampling["temperature"]
        else:
            extra_gen_cfg["temperature"] = sampling["temperature"]
    if "top_p" in sampling:
        if _HAS_NATIVE_TOP_P:
            gen_cfg["top_p"] = sampling["top_p"]
        else:
            extra_gen_cfg["top_p"] = sampling["top_p"]

    return (gen_cfg or None), (extra_gen_cfg or None)
