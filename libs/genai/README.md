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

Known Gemini chat models advertise `profile["file_mime_types"] = ["text/plain"]`.
This conservative list covers generic document inputs supported by both the
Gemini Developer API and Vertex AI. PDF, image, audio, and video support remains
represented by the dedicated profile flags. An absent list means support is
unknown, not that all file types are rejected; custom `profile` values are preserved.

The [Developer API input reference](https://ai.google.dev/api/generate-content#Blob)
lists additional text and application MIME types, but the
[Vertex AI document guide](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding)
documents `application/pdf` and `text/plain` for document input. Non-PDF documents
are processed as text, not rendered visually. File Search indexing formats are
not native chat input capabilities. Upload files separately and provide their URI
and MIME type, or send base64 file content through standard LangChain file blocks.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
