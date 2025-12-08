"""LangChain Google Generative AI integration (VertexAI).

This module contains the LangChain integrations for
[Vertex AI service](https://cloud.google.com/vertex-ai) - Google foundational models
and third-party models available on Vertex Model Garden.

**Supported integrations**

1. Other Google's foundational models: Imagen - `VertexAIImageCaptioning`,
    `VertexAIImageCaptioningChat`, `VertexAIImageEditorChat`,
    `VertexAIImageGeneratorChat`, `VertexAIVisualQnAChat`.
2. Third-party foundational models available as a an API (mdel-as-a-service) on Vertex
    Model Garden (Mistral, Llama, Anthropic) - `model_garden.ChatAnthropicVertex`,
    `model_garden_maas.VertexModelGardenLlama`,
    `model_garden_maas.VertexModelGardenMistral`.
3. Third-party foundational models deployed on Vertex AI endpoints from Vertex Model
    Garden or Huggingface - `VertexAIModelGarden`.
4. Vector Search on Vertex AI - `VectorSearchVectorStore`,
    `VectorSearchVectorStoreDatastore`, `VectorSearchVectorStoreGCS`.
5. Vertex AI evaluators for generative AI - `VertexPairWiseStringEvaluator`,
    `VertexStringEvaluator`.

You need to enable required Google Cloud APIs (depending on the integration you're
using) and set up credentials by either:

- Having credentials configured for your environment (gcloud, workload identity,
    etc...)
- Storing the path to a service account JSON file as the
    `GOOGLE_APPLICATION_CREDENTIALS` environment variable

This codebase uses the `google.auth` library which first looks for the application
credentials variable mentioned above, and then looks for system-level auth.
"""

from google.cloud.aiplatform_v1beta1.types import (
    FunctionCallingConfig,
    FunctionDeclaration,
    Schema,
    ToolConfig,
    Type,
)

from langchain_google_vertexai._enums import (
    HarmBlockThreshold,
    HarmCategory,
    Modality,
    SafetySetting,
)
from langchain_google_vertexai.chains import create_structured_runnable
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.evaluators.evaluation import (
    VertexPairWiseStringEvaluator,
    VertexStringEvaluator,
)
from langchain_google_vertexai.functions_utils import (
    PydanticFunctionsOutputParser,
)
from langchain_google_vertexai.llms import VertexAI
from langchain_google_vertexai.model_garden import VertexAIModelGarden
from langchain_google_vertexai.model_garden_maas import get_vertex_maas_model
from langchain_google_vertexai.utils import create_context_cache
from langchain_google_vertexai.vectorstores import (
    DataStoreDocumentStorage,
    GCSDocumentStorage,
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
    VectorSearchVectorStoreGCS,
)
from langchain_google_vertexai.vision_models import (
    VertexAIImageCaptioning,
    VertexAIImageCaptioningChat,
    VertexAIImageEditorChat,
    VertexAIImageGeneratorChat,
    VertexAIVisualQnAChat,
)

__all__ = [
    "ChatVertexAI",  # Deprecated
    "DataStoreDocumentStorage",
    "FunctionCallingConfig",
    "FunctionDeclaration",
    "GCSDocumentStorage",
    "HarmBlockThreshold",
    "HarmCategory",
    "Modality",
    "PydanticFunctionsOutputParser",
    "SafetySetting",
    "Schema",
    "ToolConfig",
    "Type",
    "VectorSearchVectorStore",
    "VectorSearchVectorStoreDatastore",
    "VectorSearchVectorStoreGCS",
    "VertexAI",  # Deprecated
    "VertexAIEmbeddings",  # Deprecated
    "VertexAIImageCaptioning",
    "VertexAIImageCaptioningChat",
    "VertexAIImageEditorChat",
    "VertexAIImageGeneratorChat",
    "VertexAIModelGarden",
    "VertexAIVisualQnAChat",
    "VertexPairWiseStringEvaluator",
    "VertexStringEvaluator",
    "create_context_cache",
    "create_structured_runnable",
    "get_vertex_maas_model",
]
