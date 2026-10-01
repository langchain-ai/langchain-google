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

Known Gemini chat models advertise accepted document, PDF, image, audio, and
video MIME types together through `profile["file_mime_types"]`. The legacy modality
flags remain available for compatibility. The list follows the resolved backend
and each model's input capabilities. Explicit custom profiles, including empty
dictionaries, are preserved; an absent list means file support is unknown.

For fully multimodal models, the Developer API list includes 47 types: 16 text
and application types, PDF, eight image types, 13 audio types, and nine video
types. Sources are the [input reference](https://ai.google.dev/api/generate-content#Blob),
[image guide](https://ai.google.dev/gemini-api/docs/image-understanding),
[audio guide](https://ai.google.dev/gemini-api/docs/audio), and
[video guide](https://ai.google.dev/gemini-api/docs/video-understanding).
Vertex AI advertises 29 types: plain text, PDF, three image types, 11 audio types,
and 13 video types (including aliases documented in REST examples). Sources are
its [document](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding),
[image](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding),
[audio](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding),
and [video](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding)
guides. HTML is not advertised on Vertex. Native image-generation chat models
also advertise HEIC/HEIF on Vertex, matching their model-specific input tables.
Their lists include plain text, PDF, and five image types; verified video-capable
models additionally include video types, but no audio or generic document union.
These are native chat input capabilities, not File Search indexing formats or
Live API formats. Size, duration, transport, and model-specific limits still apply.
Non-PDF documents are processed as text, not rendered visually. Upload files
separately and provide their URI and MIME type, or send base64 file content
through standard LangChain file blocks.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
