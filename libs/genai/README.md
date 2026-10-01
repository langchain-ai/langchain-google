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

## File input profiles

Gemini 3+ chat profiles expose accepted document and media MIME types through
`profile["file_mime_types"]`, filtered by the resolved backend and model modalities.
Legacy flags remain available; explicit profiles are preserved. Retired pre-3,
Live, embedding and specialized non-chat models do not advertise the list.

Formats follow the [Developer input reference](https://ai.google.dev/api/generate-content#Blob)
and Vertex [document](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding),
[image](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding),
[audio](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding)
and [video](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding)
guides. MIME support does not imply arbitrary transport or File Search support.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
