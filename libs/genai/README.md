# langchain-google-genai

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-genai?label=%20)](https://pypi.org/project/langchain-google-genai/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-genai)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-genai)](https://pypistats.org/packages/langchain-google-genai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-google-genai
```

## 📖 Documentation

View the [documentation](https://docs.langchain.com/oss/python/integrations/providers/google) for more details.

This package provides access to Google Gemini's chat, vision, embeddings, and other capabilities within the LangChain ecosystem.

---

## Overview

This package provides LangChain support for Google Gemini models (via the official [Google Generative AI SDK](https://googleapis.github.io/python-genai/)). It supports:

- Text and vision-based chat models
- Embeddings for semantic search
- Multimodal inputs and outputs
- Retrieval-Augmented Generation (RAG)
- Thought tracing with reasoning tokens
- and more!

## Quickstart

Set up your environment variable with your Gemini API key:

```bash
export GOOGLE_API_KEY=your-api-key
```

Then use the `ChatGoogleGenerativeAI` interface:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
response = llm.invoke("Sing a ballad of LangChain.")
print(response.content_blocks)
```

---

## Chat Models

See the LangChain documentation for general information about [Chat Models](https://docs.langchain.com/oss/python/langchain/models).

The main interface for the Gemini chat models is `ChatGoogleGenerativeAI`.

### Multimodal Inputs

Most Gemini models support image inputs.

```python
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?"
        },
        {
            "type": "image_url",
            "image_url": "https://picsum.photos/seed/picsum/200/300"
        },
    ]
)

response = llm.invoke([message])
print(response.content)
```

`image_url` can be:

- A public image URL
- A Google Cloud Storage path (`gcs://...`)
- A base64-encoded image (e.g., `data:image/png;base64,...`)

---

### Multimodal Outputs

Some Gemini models supports both text and inline image outputs.

```python
# Running inside a Jupyter notebook:
import base64

from IPython.display import Image, display
from langchain.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, Modality

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-image")

message = {
    "role": "user",
    "content": "Generate a photorealistic image of a cuddly cat wearing a hat.",
}

response = llm.invoke(
    [message],
    response_modalities=[Modality.TEXT, Modality.IMAGE],
)


def _get_image_base64(response: AIMessage) -> None:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]


image_base64 = _get_image_base64(response)
display(Image(data=base64.b64decode(image_base64), width=300))
```

---

### Audio Output

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-tts")

response = llm.invoke(
    "Please say The quick brown fox jumps over the lazy dog",
    generation_config=dict(response_modalities=["AUDIO"]),
)

# Base64 encoded binary data of the audio
wav_data = response.additional_kwargs.get("audio")
with open("output.wav", "wb") as f:
    f.write(wav_data)
```

---

### Multimodal Outputs in Chains

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, Modality

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image",
    response_modalities=[Modality.TEXT, Modality.IMAGE],
)

prompt = ChatPromptTemplate.from_messages([
    ("human", "Generate an image of {animal} and tell me the sound it makes.")
])

chain = {"animal": RunnablePassthrough()} | prompt | llm
response = chain.invoke("cat")
```

---

### Thinking Support

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    thinking_budget=1024
)

response = llm.invoke("How many O's are in Google? How did you verify your answer?")
reasoning_score = response.usage_metadata["output_token_details"]["reasoning"]

print("Response:", response.content)
print("Reasoning tokens used:", reasoning_score)
```

---

## Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("hello, world!")
print(vector)
```

---

## Semantic Retrieval (RAG)

Use Gemini with RAG to retrieve relevant documents from your knowledge base.

```python
from langchain_google_genai.vectorstores import GoogleVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

# Create a corpus (collection of documents)
corpus_store = GoogleVectorStore.create_corpus(display_name="My Corpus")

# Create a document under that corpus
document_store = GoogleVectorStore.create_document(
    corpus_id=corpus_store.corpus_id, display_name="My Document"
)

# Load and upload documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
for file in DirectoryLoader(path="data/").load():
    chunks = text_splitter.split_documents([file])
    document_store.add_documents(chunks)

# Query the document corpus
aqa = corpus_store.as_aqa()
response = aqa.invoke("What is the meaning of life?")

print("Answer:", response.answer)
print("Passages:", response.attributed_passages)
print("Answerable probability:", response.answerable_probability)
```

---

## Resources

- [Gemini Model Documentation](https://ai.google.dev/gemini-api/docs/models)
- [Google Generative AI SDK](https://googleapis.github.io/python-genai/)
