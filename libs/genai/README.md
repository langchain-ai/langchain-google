# langchain-google-genai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-genai?label=%20)](https://pypi.org/project/langchain-google-genai/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-genai)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-genai)](https://pypistats.org/packages/langchain-google-genai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

This package provides access to Google Gemini's chat, vision, embeddings, and other capabilities within the LangChain ecosystem.

## Quick Install

```bash
pip install langchain-google-genai
```

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_google_genai/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/google#google-generative-ai).

### Streaming tool call arguments (Gemini 3+)

Gemini 3 Pro and later can stream tool/function call arguments as typed leaf updates (`PartialArg`) instead of one final block. Opt in by setting `stream_function_call_arguments=True`; the leaf updates are then surfaced on each `AIMessageChunk` via `additional_kwargs["gemini_partial_args"]`, so consumers can preview values as they arrive without waiting for the full JSON to close.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    stream_function_call_arguments=True,
).bind_tools([book_flight])

async for chunk in llm.astream("Book SFO -> JFK on 2026-06-01."):
    for delta in chunk.additional_kwargs.get("gemini_partial_args", []):
        # delta = {"json_path": "$.date", "value": "2026-06-01",
        #          "will_continue": True, "tool_call_id": ..., ...}
        print(delta["json_path"], "=", delta["value"])
```

Each entry carries the resolved `tool_call_id`, the raw SDK `FunctionCall.id` (`sdk_call_id`, may be `None` in current Vertex preview), the `json_path` (e.g. `$.date`), the typed scalar `value`, and `will_continue`. The standard `tool_call_chunks` continue to carry the SDK-assembled `fc.args` as a single atomic seal chunk — merging chunks with `+` still yields the complete `tool_calls` list as today.

#### Caveats

- **Vertex AI backend only.** The Gemini API (`generativelanguage.googleapis.com`) backend rejects this flag at the SDK serializer (`google.genai` raises `ValueError` before any wire call), so it is only useful with `vertexai=True`.
- The flag is only sent on the streaming endpoint. Non-streaming `invoke` / `ainvoke` calls ignore it because the Vertex `:generateContent` endpoint rejects the option (see [vercel/ai#14314](https://github.com/vercel/ai/issues/14314), [vercel/ai#14352](https://github.com/vercel/ai/pull/14352)).
- Translating `PartialArg` deltas into incremental `tool_call_chunks` (the full Anthropic-/OpenAI-style streaming experience) is intentionally out of scope for this opt-in: the Vertex 3.x preview wire format omits `FunctionCall.id` and chunks string values mid-leaf, so a faithful translator needs additional design — tracked as a follow-up. The structured side channel above is sufficient for most preview/UI use cases.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
