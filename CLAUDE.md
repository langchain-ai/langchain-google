# Global development guidelines for the LangChain monorepo

This document provides context to understand the LangChain Python project and assist with development.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages that use `uv`.

### Development tools & commands**

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `make` – Task runner for common development commands. Feel free to look at the `Makefile` for available commands and usage patterns.
- `ruff` – Fast Python linter and formatter
- `mypy` – Static type checking
- `pytest` – Testing framework

This monorepo uses `uv` for dependency management. Local development uses editable installs: `[tool.uv.sources]`

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

#### Key config files

- pyproject.toml: Main workspace configuration with dependency groups
- uv.lock: Locked dependencies for reproducible builds
- Makefile: Development tasks

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Refer to .github/workflows/pr_lint for allowed types and scopes.

#### Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the known_users set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- The testing file structure should mirror the source code structure.

**Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)
- [ ] Does the test suite fail if your new logic is broken?

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")

## Additional resources

- **Documentation:** <https://docs.langchain.com/oss/python/langchain/overview> and source at <https://github.com/langchain-ai/docs> or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [`.github/CONTRIBUTING.md`](https://docs.langchain.com/oss/python/contributing/overview)

# Google-specific instructions

You can find the official SDK documentation and code samples here:
<https://ai.google.dev/gemini-api/docs>

## Golden rule: use the current SDK

- **Library:** Google GenAI SDK
- **Python package:** `google-genai`
- **Legacy libraries**: (`google-generativeai` and `google-ai-generativelanguage`) are deprecated.

**APIs and usage:**

- **Incorrect:** `import google.generativeai as genai` -> **Correct:** `from google import genai`
- **Incorrect:** `from google.ai import generativelanguage_v1`  -> **Correct:** `from google import genai`
- **Incorrect:** `from google.generativeai` -> **Correct:** `from google import genai`
- **Incorrect:** `from google.generativeai import types` -> **Correct:** `from google.genai import types`
- **Incorrect:** `import google.generativeai as genai` -> **Correct:** `from google import genai`
- **Incorrect:** `genai.configure(api_key=...)` -> **Correct:** `client = genai.Client(api_key='...')`
- **Incorrect:** `model = genai.GenerativeModel(...)`
- **Incorrect:** `model.generate_content(...)` -> **Correct:** `client.models.generate_content(...)`
- **Incorrect:** `response = model.generate_content(..., stream=True)` -> **Correct:** `client.models.generate_content_stream(...)`
- **Incorrect:** `genai.GenerationConfig(...)` -> **Correct:** `types.GenerateContentConfig(...)`
- **Incorrect:** `safety_settings={...}` -> **Correct:** Use `safety_settings` inside a `GenerateContentConfig` object.
- **Incorrect:** `from google.api_core.exceptions import GoogleAPIError` -> **Correct:** `from google.genai.errors import APIError`
- **Incorrect:** `types.ResponseModality.TEXT`

## Initialization and API key

The `google-genai` library requires creating a client object for all API calls.

## Models

- By default, use the following models when using `google-genai`:
  - **General Text & Multimodal Tasks:** `gemini-2.5-flash`
  - **Coding and Complex Reasoning Tasks:** `gemini-2.5-pro`
  - **Low Latency & High Volume Tasks:** `gemini-2.5-flash-lite`
  - **Image Editing and Manipulation:** `gemini-2.5-flash-image`
  - **High-Quality Image Generation:** `imagen-4.0-generate-001`
  - **Rapid Image Generation:** `imagen-4.0-fast-generate-001`
  - **Advanced Image Generation:** `imagen-4.0-ultra-generate-001`
  - **High-Fidelity Video Generation:** `veo-3.0-generate-001` or `veo-3.1-generate-preview`
  - **Fast Video Generation:** `veo-3.0-fast-generate-001` or `veo-3.1-fast-generate-preview`
  - **Advanced Video Editing Tasks:** `veo-3.1-generate-preview`

- Do not use the following deprecated models (or their variants like `gemini-1.5-flash-latest`):
  - **Prohibited:** `gemini-1.5-flash`
  - **Prohibited:** `gemini-1.5-pro`
  - **Prohibited:** `gemini-pro`

## Basic inference

Here's how to generate a response from a text prompt.

```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='why is the sky blue?',
)

print(response.text) # output is often markdown
```

Multimodal inputs are supported by passing a PIL Image in the `contents` list:

```python
from google import genai
from PIL import Image

client = genai.Client()

image = Image.open(img_path)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[image, 'explain that image'],
)

print(response.text) # The output often is markdown
```

You can also use `Part.from_bytes` type to pass a variety of data types (images,
audio, video, pdf).

```python
from google.genai import types

with open('path/to/small-sample.jpg', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        'Caption this image.'
    ]
)

print(response.text)
```

For larger files, use `client.files.upload`:

```python
f = client.files.upload(file=img_path)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[f, 'can you describe this image?']
)
```

You can delete files after use like this:

```python
myfile = client.files.upload(file='path/to/sample.mp3')
client.files.delete(name=myfile.name)
```

## Additional capabilities and configurations

Below are examples of advanced configurations.

### Thinking

Gemini 2.5 series models and above support thinking, which is on by default for `gemini-2.5-flash`. It can be adjusted by using `thinking_budget` setting. Setting it to zero turns thinking off, and will reduce latency.

```python
from google import genai
from google.genai import types

client = genai.Client()

client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is AI?',
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )
    )
)
```

IMPORTANT NOTES:

- Minimum thinking budget for `gemini-2.5-pro` is `128` and thinking can not be turned off for that model.
- No models (apart from Gemini 2.5 series) support thinking or thinking budgets APIs. Do not try to adjust thinking budgets other models (such as `gemini-2.0-flash` or `gemini-2.0-pro`) otherwise it will cause syntax errors.

### System instructions

Use system instructions to guide model's behavior.

```python
from google import genai
from google.genai import types

client = genai.Client()

config = types.GenerateContentConfig(
    system_instruction='You are a pirate',
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=config,
)

print(response.text)
```

### Hyperparameters

You can also set `temperature` or `max_output_tokens` within `types.GenerateContentConfig`

**Avoid** setting `max_output_tokens`, `topP`, `topK` unless explicitly requested by the user.

### Safety configurations

Avoid setting safety configurations unless explicitly requested by the user. If explicitly asked for by the user, here is a sample API:

```python
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client()

img = Image.open('/path/to/img')
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=['Do these look store-bought or homemade?', img],
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ]
    )
)

print(response.text)
```

### Streaming

It is possible to stream responses to reduce user perceived latency:

```python
from google import genai

client = genai.Client()

response = client.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents=['Explain how AI works']
)
for chunk in response:
    print(chunk.text, end='')
```

### Chat

For multi-turn conversations, use the `chats` service to maintain conversation history.

```python
from google import genai

client = genai.Client()
chat = client.chats.create(model='gemini-2.5-flash')

response = chat.send_message('I have 2 dogs in my house.')
print(response.text)

response = chat.send_message('How many paws are in my house?')
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}', end=': ')
    print(message.parts[0].text)
```

### Structured outputs

Use structured outputs to force the model to return a response that conforms to a specific Pydantic schema.

```python
from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client()

# Define the desired output structure using Pydantic
class Recipe(BaseModel):
    recipe_name: str
    description: str
    ingredients: list[str]
    steps: list[str]

# Request the model to populate the schema
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Provide a classic recipe for chocolate chip cookies.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_json_schema=Recipe.model_json_schema(),
    ),
)

# The response.text will be a valid JSON string matching the Recipe schema
print(response.text)
```

#### Function calling (Tools)

You can provide the model with tools (functions) it can use to bring in external information to answer a question or act on a request outside the model.

```python
from google import genai
from google.genai import types

client = genai.Client()

# Define a function that the model can call (to access external information)
def get_current_weather(city: str) -> str:
    """Returns the current weather in a given city. For this example, it's hardcoded."""
    if 'boston' in city.lower():
        return 'The weather in Boston is 15°C and sunny.'
    else:
        return f'Weather data for {city} is not available.'

# Make the function available to the model as a tool
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather]
    ),
)
# The model may respond with a request to call the function
if response.function_calls:
    print('Function calls requested by the model:')
    for function_call in response.function_calls:
        print(f'- Function: {function_call.name}')
        print(f'- Args: {dict(function_call.args)}')
else:
    print('The model responded directly:')
    print(response.text)
```

### Generate images

Here's how to generate images using the Imagen models. Start with the fast model as it should cover most use-cases, and move to the more standard or the ultra models for advanced use-cases.

```python
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client()

result = client.models.generate_images(
    model='imagen-4.0-fast-generate-001',
    prompt='Image of a cat',
    config=types.GenerateImagesConfig(
        number_of_images=1, # 1 to 4 (always 1 for the ultra model)
        output_mime_type='image/jpeg',
        person_generation='ALLOW_ADULT', # 'ALLOW_ALL' (but not in Europe/Mena), 'DONT_ALLOW' or 'ALLOW_ADULT'
        aspect_ratio='1:1' # '1:1', '3:4', '4:3', '9:16', or '16:9'
    )
)

for generated_image in result.generated_images:
    image = Image.open(BytesIO(generated_image.image.image_bytes))
```

### Edit images

Editing images is better done using the Gemini native image generation model, and it is recommended to use chat mode. Configs are not supported in this model (except modality).

```python
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = """
Create a picture of my cat eating a nano-banana in a fancy restaurant under the gemini constellation
"""
image = Image.open('/path/to/image.png')

# Create the chat
chat = client.chats.create(model='gemini-2.5-flash-image')
# Send the image and ask for it to be edited
response = chat.send_message([prompt, image])

# Get the text and the image generated
for i, part in enumerate(response.candidates[0].content.parts):
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save(f'generated_image_{i}.png') # Multiple images can be generated

# Continue iterating
chat.send_message('Can you make it a bananas foster?')
```

### Generate videos

Here's how to generate videos using the Veo models. Usage of Veo can be costly, so after generating code for it, give user a heads up to check pricing for Veo. Start with the fast model since the result quality is usually sufficient, and swap to the larger model if needed.

```python
import time
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client()

image = Image.open('path/to/image.png') # Optional

operation = client.models.generate_videos(
    model='veo-3.0-fast-generate-001',
    prompt='Panning wide shot of a calico kitten sleeping in the sunshine',
    image=image,
    config=types.GenerateVideosConfig(
        person_generation='dont_allow',  # 'dont_allow' or 'allow_adult'
        aspect_ratio='16:9',  # '16:9' or '9:16'
        number_of_videos=1, # supported value is 1-4, use 1 by default
        duration_seconds=8, # supported value is 5-8
    ),
)

while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

for n, generated_video in enumerate(operation.response.generated_videos):
    client.files.download(file=generated_video.video) # just file=, no need for path= as it doesn't save yet
    generated_video.video.save(f'video{n}.mp4')  # saves the video
```

### Search grounding

Google Search can be used as a tool for grounding queries that with up to date information from the web.

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What was the score of the latest Olympique Lyonais game?',
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(google_search=types.GoogleSearch())
        ]
    ),
)

# Response
print(f'Response:\n {response.text}')
# Search details
print(f'Search Query: {response.candidates[0].grounding_metadata.web_search_queries}')
# Urls used for grounding
print(f"Search Pages: {', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])}")
```

The output `response.text` will likely not be in JSON format, do not attempt to parse it as JSON.

### Content and part hierarchy

While the simpler API call is often sufficient, you may run into scenarios where you need to work directly with the underlying `Content` and `Part` objects for more explicit control. These are the fundamental building blocks of the `generate_content` API.

For instance, the following simple API call:

```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='How does AI work?'
)
print(response.text)
```

is effectively a shorthand for this more explicit structure:

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Content(role='user', parts=[types.Part.from_text(text='How does AI work?')]),
    ]
)
print(response.text)
```

## Other APIs

The list of APIs and capabilities above are not comprehensive. If users ask you to generate code for a capability not provided above, refer them to ai.google.dev/gemini-api/docs.

## Running tests

If Vertex tests fail due to expired GCP credentials, remind the tester to re-authenticate: `gcloud auth application-default login`

## Useful links

- Documentation: ai.google.dev/gemini-api/docs
- API Keys and Authentication: ai.google.dev/gemini-api/docs/api-key
- Models: ai.google.dev/models
- API Pricing: ai.google.dev/pricing
- Rate Limits: ai.google.dev/rate-limits
