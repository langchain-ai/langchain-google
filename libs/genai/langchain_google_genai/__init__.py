"""LangChain Google Gen AI integration.

!!! note "Vertex AI Consolidated"

    As of `langchain-google-genai 4.0.0`, this package uses the
    [consolidated Google Gen AI SDK](https://googleapis.github.io/python-genai/)
    instead of the legacy [`google-ai-generativelanguage`](https://googleapis.dev/python/generativelanguage/latest/)
    SDK.

    This brings support for Gemini models both via the Gemini API and Gemini API in
    Vertex AI, superseding `langchain-google-vertexai`. Users should migrate to this
    package for continued support of Google's Generative AI models.

    Certain Vertex AI features are not yet supported in the consolidated SDK (and
    subsequently this package); see [the docs](https://docs.langchain.com/oss/python/integrations/providers/google)
    for more details.

This module provides an interface to Google's Generative AI models, specifically the
Gemini series, with the LangChain framework. It provides classes for interacting with
chat models, generating embeddings, and more.

**Chat Models**

The `ChatGoogleGenerativeAI` class is the primary interface for interacting with
Google's Gemini chat models. It allows users to send and receive messages using a
specified Gemini model, suitable for various conversational AI applications.

**Embeddings**

The `GoogleGenerativeAIEmbeddings` class provides functionalities to generate embeddings
using Google's models. These embeddings can be used for a range of NLP tasks, including
semantic analysis, similarity comparisons, and more.

See [the docs](https://docs.langchain.com/oss/python/integrations/providers/google) for
more information on usage of this package.
"""

from langchain_google_genai._enums import (
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.google_vector_store import (
    DoesNotExistsException,
    GoogleVectorStore,
)
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.utils import create_context_cache

__all__ = [
    "ChatGoogleGenerativeAI",
    "DoesNotExistsException",
    "GoogleGenerativeAI",
    "GoogleGenerativeAIEmbeddings",
    "GoogleVectorStore",
    "HarmBlockThreshold",
    "HarmCategory",
    "MediaResolution",
    "Modality",
    "create_context_cache",
]
