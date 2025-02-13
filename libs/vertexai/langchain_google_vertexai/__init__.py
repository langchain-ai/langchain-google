"""**LangChain Google Generative AI Integration**

This module contains the LangChain integrations for Vertex AI service - Google foundational models, third-party foundational modela available on Vertex Model Garden and.

**Supported integrations**

1. Google's founational models: Gemini family, Codey, embeddings - `ChatVertexAI`, `VertexAI`, `VertexAIEmbeddings`.
2. Other Google's foundational models: Imagen - `VertexAIImageCaptioning`, `VertexAIImageCaptioningChat`, `VertexAIImageEditorChat`, `VertexAIImageGeneratorChat`, `VertexAIVisualQnAChat`.
3. Third-party foundational models available as a an API (mdel-as-a-service) on Vertex Model Garden (Mistral, Llama, Anthropic) - `model_garden.ChatAnthropicVertex`, `model_garden_maas.VertexModelGardenLlama`, `model_garden_maas.VertexModelGardenMistral`.
4. Third-party foundational models deployed on Vertex AI endpoints from Vertex Model Garden or Hugginface - `VertexAIModelGarden`.
5. Gemma deployed on Vertex AI endpoints or locally - `GemmaChatLocalHF`, `GemmaChatLocalKaggle`, `GemmaChatVertexAIModelGarden`, `GemmaLocalHF`, `GemmaLocalKaggle`, `GemmaVertexAIModelGarden`.
5. Vector Search on Vertex AI - `VectorSearchVectorStore`, `VectorSearchVectorStoreDatastore`, `VectorSearchVectorStoreGCS`.
6. Vertex AI evaluators for generative AI - `VertexPairWiseStringEvaluator`, `VertexStringEvaluator`.

Take a look at detailed documentation for each class for further details.

**Installation**

```bash
pip install -U langchain-google-vertexai
```

You need to enable needed Google Cloud APIs (depending on the integration you're using) and set up credentials by either:
    - Have credentials configured for your environment
        (gcloud, workload identity, etc...)
    - Store the path to a service account JSON file as the
        GOOGLE_APPLICATION_CREDENTIALS environment variable

This codebase uses the google.auth library which first looks for the application
credentials variable mentioned above, and then looks for system-level auth.

For more information, see:
https://cloud.google.com/docs/authentication/application-default-credentials#GAC
and 
https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth.


"""  # noqa: E501

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
from langchain_google_vertexai.gemma import (
    GemmaChatLocalHF,
    GemmaChatLocalKaggle,
    GemmaChatVertexAIModelGarden,
    GemmaLocalHF,
    GemmaLocalKaggle,
    GemmaVertexAIModelGarden,
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
    "ChatVertexAI",
    "create_structured_runnable",
    "DataStoreDocumentStorage",
    "FunctionCallingConfig",
    "FunctionDeclaration",
    "GCSDocumentStorage",
    "GemmaChatLocalHF",
    "GemmaChatLocalKaggle",
    "GemmaChatVertexAIModelGarden",
    "GemmaLocalHF",
    "GemmaLocalKaggle",
    "GemmaVertexAIModelGarden",
    "HarmBlockThreshold",
    "HarmCategory",
    "PydanticFunctionsOutputParser",
    "SafetySetting",
    "Schema",
    "ToolConfig",
    "Type",
    "VectorSearchVectorStore",
    "VectorSearchVectorStoreDatastore",
    "VectorSearchVectorStoreGCS",
    "VertexAI",
    "VertexAIEmbeddings",
    "VertexAIImageCaptioning",
    "VertexAIImageCaptioningChat",
    "VertexAIImageEditorChat",
    "VertexAIImageGeneratorChat",
    "VertexAIModelGarden",
    "VertexAIVisualQnAChat",
    "VertexPairWiseStringEvaluator",
    "VertexStringEvaluator",
    "create_context_cache",
    "get_vertex_maas_model",
]
