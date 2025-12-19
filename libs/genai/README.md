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

## ⚡️ Quick Start

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# export GOOGLE_API_KEY="your-api-key"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fast, cost-effective, and capable model
    temperature=0,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
# Example output: J'adore la programmation.
```

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_google_genai/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/google#google-generative-ai).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
