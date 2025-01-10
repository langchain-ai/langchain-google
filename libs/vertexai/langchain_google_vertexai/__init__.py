"""
## langchain-google-vertexai

This module contains the LangChain integrations for Google Cloud generative models.

## Installation

```bash
pip install -U langchain-google-vertexai
```

## Supported Models (MaaS: Model-as-a-Service)

1. Llama 
2. Mistral
3. Gemma
4. Vision models

Integration on Google Cloud Vertex AI Model-as-a-Service.

For more information, see:
    https://cloud.google.com/blog/products/ai-machine-learning/llama-3-1-on-vertex-ai

#### Setup

You need to enable a corresponding MaaS model (Google Cloud UI console ->
Vertex AI -> Model Garden -> search for a model you need and click enable)

You must have the langchain-google-vertexai Python package installed
.. code-block:: bash

    pip install -U langchain-google-vertexai

And either:
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

## Chat Language Model vs Base Language Models

A "base model" is a large language model trained on a wide variety of text data, 
providing a general-purpose foundation for various tasks.

While a "chat model" is a specialized version of a base model specifically fine-tuned 
to excel in conversational interactions, meaning it's optimized to understand and 
respond to dialogue within a chat context, often with features like maintaining 
conversational context and generating engaging responses.

## Gemma models

Gemma is a family of lightweight, state-of-the-art open models built from the same 
research and technology used to create the Gemini models. Developed by Google DeepMind 
and other teams across Google.

We currently support the following Gemma models.

1. GemmaChatLocalHF: Gemma Chat local model from Hugging Face.
2. GemmaChatLocalKaggle - Gemma chat local model from Kaggle.
3. GemmaLocalHF - Gemma Local model from Hugging Face
4. GemmaLocalKaggle - Gemma Local model from Kaggle

## Vision models

### Image Captioning

Generate Captions from Image. Implementation of the Image Captioning model 

The model takes a list of prompts as an input where each prompt must be a 
string that represents an image. Currently supported are:

1. Google Cloud Storage URI 
2. B64 encoded string
3. Local file path
4. Remote url

The model returns Captions generated from every prompt (Image).

### Image Generation

Generates images from text prompt. 

The message must be a list of only one element with one part i.e. the user prompt.

### Visual QnA Chat

Answers questions about an image. Chat implementation of a visual QnA model.

The model takes a list of messages. The first message should contain a string 
representation of the image. Currently supported are:

1. Google Cloud Storage URI
2. B64 encoded string
3. Local file path
4. Remote url

There has to be at least another message with the first question.
The model returns the generated answer for the questions asked.

### Image Editor

Given an image and a prompt, edit the image. Currently only supports mask free editing.

The message must be a list of only one element with two part:

1. The image as a dictionary: { 'type': 'image_url', 'image_url': {'url': <message_str>} }
2. The user prompt.

## Chat Models

`ChatVertexAI` class exposes models such as `gemini-pro` and `chat-bison`.

To use, you should have Google Cloud project with APIs enabled, and configured 
credentials. Initialize the model as:

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

## Vector Stores

#### Vector Search Vector Store GCS

VertexAI VectorStore that handles the search and indexing using Vector Search 
and stores the documents in Google Cloud Storage.

#### Vector Search Vector Store Datastore

VectorSearch with DatasTore document storage.
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
