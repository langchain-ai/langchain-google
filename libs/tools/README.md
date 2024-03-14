# langchain-google-genai

This package contains the LangChain integrations for Gemini through their generative-ai SDK.

## Installation

```bash
pip install -U langchain-google-genai
```

### Image utilities
To use image utility methods, like loading images from GCS urls, install with extras group 'images':

```bash
pip install -e "langchain-google-genai[images]"
```

## Chat Models

This package contains the `ChatGoogleGenerativeAI` class, which is the recommended way to interface with the Google Gemini series of models.

To use, install the requirements, and configure your environment.

```bash
export GOOGLE_API_KEY=your-api-key
```

Then initialize

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

#### Multimodal inputs

Gemini vision model supports image inputs when providing a single chat message. Example:

```
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
llm.invoke([message])
```

The value of `image_url` can be any of the following:

- A public image URL
- An accessible gcs file (e.g., "gcs://path/to/file.png")
- A local file path
- A base64 encoded image (e.g., `data:image/png;base64,abcd124`)
- A PIL image



## Embeddings

This package also adds support for google's embeddings models.

```
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embeddings.embed_query("hello, world!")
```

## Semantic Retrieval

Enables retrieval augmented generation (RAG) in your application.

```
# Create a new store for housing your documents.
corpus_store = GoogleVectorStore.create_corpus(display_name="My Corpus")

# Create a new document under the above corpus.
document_store = GoogleVectorStore.create_document(
    corpus_id=corpus_store.corpus_id, display_name="My Document"
)

# Upload some texts to the document.
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
for file in DirectoryLoader(path="data/").load():
    documents = text_splitter.split_documents([file])
    document_store.add_documents(documents)

# Talk to your entire corpus with possibly many documents. 
aqa = corpus_store.as_aqa()
answer = aqa.invoke("What is the meaning of life?")

# Read the response along with the attributed passages and answerability.
print(response.answer)
print(response.attributed_passages)
print(response.answerable_probability)
```
