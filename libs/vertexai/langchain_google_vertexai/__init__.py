"""
## langchain-google-vertexai

This module contains the LangChain integrations for Google Cloud generative models.

## Installation

```bash
pip install -U langchain-google-vertexai
```

## Chat Models

`ChatVertexAI` class exposes models such as `gemini-pro` and `chat-bison`.

To use, you should have Google Cloud project with APIs enabled, and configured credentials. Initialize the model as:

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

You can use other models, e.g. `chat-bison`:

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="chat-bison", temperature=0.3)
llm.invoke("Sing a ballad of LangChain.")
```

#### Multimodal inputs

Gemini vision model supports image inputs when providing a single chat message. Example:

```python
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro-vision")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/picsum/200/300"}},
    ]
)
llm.invoke([message])
```

The value of `image_url` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A base64 encoded image (e.g., `data:image/png;base64,abcd124`)

## Embeddings

You can use Google Cloud's embeddings models as:

```python
from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()
embeddings.embed_query("hello, world!")
```

## LLMs

You can use Google Cloud's generative AI models as Langchain LLMs:

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

template = \"""Question: {question}

Answer: Let's think step by step.\"""
prompt = PromptTemplate.from_template(template)

llm = ChatVertexAI(model_name="gemini-pro")
chain = prompt | llm

question = "Who was the president of the USA in 1994?"
print(chain.invoke({"question": question}))
```

You can use Gemini and Palm models, including code-generations ones:

```python

from langchain_google_vertexai import VertexAI

llm = VertexAI(model_name="code-bison", max_output_tokens=1000, temperature=0.3)

question = "Write a python function that checks if a string is a valid email address"

output = llm(question)
```
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
