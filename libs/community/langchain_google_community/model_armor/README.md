# Model Armor Runnables for LangChain

This package provides LangChain-compatible Runnables for prompt and response sanitization using Google Cloud Model Armor.


## Usage Example

```python
from google.cloud.modelarmor_v1 import ModelArmorClient
from langchain_google_community.model_armor import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

# Initialize Model Armor client and template ID
gcp_client = ModelArmorClient()
template_id = "projects/<project>/locations/<location>/templates/<template_id>"

# Create Runnables
prompt_sanitizer = ModelArmorSanitizePromptRunnable(
    client=gcp_client,
    template_id=template_id,
    fail_open=True,
    return_findings=True,
)

response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client=gcp_client,
    template_id=template_id,
    fail_open=True,
    return_findings=True,
)

# Sanitize a prompt
result = prompt_sanitizer.invoke("Your prompt here")
print(result)

# Sanitize a response
result = response_sanitizer.invoke("LLM response here")
print(result)
```

## Using Model Armor Runnables in a Chain

You can use the Model Armor Runnables as part of a LangChain Runnable sequence or chain:

```python
from langchain_core.runnables import RunnableSequence

# Example: Chain prompt sanitizer, LLM, and response sanitizer
# (Assume `llm_runnable` is your LLM Runnable)

chain = RunnableSequence([
    prompt_sanitizer,
    llm_runnable,  # Your LLM or other processing step
    response_sanitizer,
])

# Run the chain
output = chain.invoke("Your prompt here")
print(output)
```

## Features
- Prompt and response sanitization using Google Cloud Model Armor
- Configurable fail-open to continue flow incase of unsafe prompt/response
- Configurable return object to include Model Armor sanitization findings.
- Compatible with LangChain Runnable interface
- Easy integration into LangChain chains and pipelines

## License
See LICENSE file.
