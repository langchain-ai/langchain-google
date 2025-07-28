# langchain-google-vertexai

This package contains the LangChain integrations for Google Cloud generative models.

## Contents

1. [Installation](#installation)
2. [Chat Models](#chat-models)
   * [Multimodal inputs](#multimodal-inputs)
3. [Embeddings](#embeddings)
4. [LLMs](#llms)
5. [Code Generation](#code-generation)
   * [Example: Generate a Python function](#example-generate-a-python-function)
   * [Example: Generate JavaScript code](#example-generate-javascript-code)
   * [Notes](#notes)

## Installation

```bash
pip install -U langchain-google-vertexai
```

## Chat Models

`ChatVertexAI` class exposes models such as `gemini-pro` and other Gemini variants.

To use, you should have a Google Cloud project with APIs enabled, and configured credentials. Initialize the model as:

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")
```

### Multimodal inputs

Gemini supports image inputs when providing a single chat message. Example:

```python
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-2.0-flash-001")
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/picsum/200/300"}},
    ]
)
llm.invoke([message])
```

The value of `image_url` can be:

* A public image URL
* An accessible Google Cloud Storage (GCS) file (e.g., `"gcs://path/to/file.png"`)
* A base64 encoded image (e.g., `"data:image/png;base64,abcd124"`)

## Embeddings

Google Cloud embeddings models can be used as:

```python
from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()
embeddings.embed_query("hello, world!")
```

## LLMs

Use Google Cloud's generative AI models as LangChain LLMs:

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm = ChatVertexAI(model_name="gemini-pro")
chain = prompt | llm

question = "Who was the president of the USA in 1994?"
print(chain.invoke({"question": question}))
```

## Code Generation

You can use Gemini models for code generation tasks to generate code snippets, functions, or scripts in various programming languages.

### Example: Generate a Python function

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro", temperature=0.3, max_output_tokens=1000)

prompt = "Write a Python function that checks if a string is a valid email address."

generated_code = llm.invoke(prompt)
print(generated_code)
```

### Example: Generate JavaScript code

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-pro", temperature=0.3, max_output_tokens=1000)
prompt_js = "Write a JavaScript function that returns the factorial of a number."

print(llm.invoke(prompt_js))
```

### Notes

* Adjust `temperature` to control creativity (higher values increase randomness).
* Use `max_output_tokens` to limit the length of the generated code.
* Gemini models are well-suited for code generation tasks with advanced understanding of programming concepts.