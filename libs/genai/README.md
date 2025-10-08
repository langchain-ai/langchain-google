# langchain-google-genai

**LangChain integration for Google Gemini models using the `generative-ai` SDK**

This package enables seamless access to Google Gemini's chat, vision, embeddings, and retrieval-augmented generation (RAG) features within the LangChain ecosystem.

---

## Table of Contents

- [langchain-google-genai](#langchain-google-genai)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Chat Models](#chat-models)
    - [Multimodal Inputs](#multimodal-inputs)
    - [Multimodal Outputs](#multimodal-outputs)
    - [Audio Output](#audio-output)
    - [Multimodal Outputs in Chains](#multimodal-outputs-in-chains)
    - [Thinking Support](#thinking-support)
  - [Embeddings](#embeddings)
  - [Semantic Retrieval (RAG)](#semantic-retrieval-rag)
  - [Resources](#resources)

---

## Overview

This package provides LangChain support for Google Gemini models (via the official [Google Generative AI SDK](https://googleapis.github.io/python-genai/)). It supports:

- Text and vision-based chat models
- Embeddings for semantic search
- Multimodal inputs and outputs
- Retrieval-Augmented Generation (RAG)
- Thought tracing with reasoning tokens

---

## Installation

```bash
pip install -U langchain-google-genai

# or, with uv:
uv add langchain-google-genai
````

---

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
print(response.content)
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
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image-preview")

response = llm.invoke(
    "Generate an image of a cat and say meow",
    generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
)

image_base64 = response.content[1].get("image_url").get("url").split(",")[-1]
meow_text = response.content[0]
print(meow_text)
# In Jupyter, display the image:
from base64 import b64decode
from IPython.display import Image, display

img_bytes = b64decode(image_base64)
display(Image(data=img_bytes))
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
    model="gemini-2.5-flash-image-preview",
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

- [LangChain Documentation](https://docs.langchain.com/)
- [Google Generative AI SDK](https://googleapis.github.io/python-genai/)
- [Gemini Model Documentation](https://ai.google.dev/gemini-api/docs)
