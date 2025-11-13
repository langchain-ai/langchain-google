# Global Development Guidelines for LangChain Projects

## Core Development Principles

### 1. Maintain Stable Public Interfaces ⚠️ CRITICAL

**Always attempt to preserve function signatures, argument positions, and names for exported/public methods.**

❌ **Bad - Breaking Change:**

```python
def get_user(id, verbose=False):  # Changed from `user_id`
    pass
```

✅ **Good - Stable Interface:**

```python
def get_user(user_id: str, verbose: bool = False) -> User:
    """Retrieve user by ID with optional verbose output."""
    pass
```

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

🧠 *Ask yourself:* "Would this change break someone's code if they used it last week?"

### 2. Code Quality Standards

**All Python code MUST include type hints and return types.**

❌ **Bad:**

```python
def p(u, d):
    return [x for x in u if x not in d]
```

✅ **Good:**

```python
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Filter out users that are not in the known users set.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the known_users set.
    """
    return [user for user in users if user not in known_users]
```

**Style Requirements:**

- Use descriptive, **self-explanatory variable names**. Avoid overly short or cryptic identifiers.
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense
- Avoid unnecessary abstraction or premature optimization
- Follow existing patterns in the codebase you're modifying

### 3. Testing Requirements

**Every new feature or bugfix MUST be covered by unit tests.**

**Test Organization:**

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- Use `pytest` as the testing framework

**Test Quality Checklist:**

- [ ] Tests fail when your new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases and error conditions are tested
- [ ] Use fixtures/mocks for external dependencies
- [ ] Tests are deterministic (no flaky tests)

Checklist questions:

- [ ] Does the test suite fail if your new logic is broken?
- [ ] Are all expected behaviors exercised (happy path, invalid input, etc)?
- [ ] Do tests use fixtures or mocks where needed?

```python
def test_filter_unknown_users():
    """Test filtering unknown users from a list."""
    users = ["alice", "bob", "charlie"]
    known_users = {"alice", "bob"}

    result = filter_unknown_users(users, known_users)

    assert result == ["charlie"]
    assert len(result) == 1
```

### 4. Security and Risk Assessment

**Security Checklist:**

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

❌ **Bad:**

```python
def load_config(path):
    with open(path) as f:
        return eval(f.read())  # ⚠️ Never eval config
```

✅ **Good:**

```python
import json

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
```

### 5. Documentation Standards

**Use Google-style docstrings with Args section for all public functions.**

❌ **Insufficient Documentation:**

```python
def send_email(to, msg):
    """Send an email to a recipient."""
```

✅ **Complete Documentation:**

```python
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """
    Send an email to a recipient with specified priority.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level (`'low'`, `'normal'`, `'high'`).

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        `InvalidEmailError`: If the email address format is invalid.
        `SMTPConnectionError`: If unable to connect to email server.
    """
```

**Documentation Guidelines:**

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")

📌 *Tip:* Keep descriptions concise but clear. Only document return values if non-obvious.

### 6. Architectural Improvements

**When you encounter code that could be improved, suggest better designs:**

❌ **Poor Design:**

```python
def process_data(data, db_conn, email_client, logger):
    # Function doing too many things
    validated = validate_data(data)
    result = db_conn.save(validated)
    email_client.send_notification(result)
    logger.log(f"Processed {len(data)} items")
    return result
```

✅ **Better Design:**

```python
@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    items_processed: int
    success: bool
    errors: List[str] = field(default_factory=list)

class DataProcessor:
    """Handles data validation, storage, and notification."""

    def __init__(self, db_conn: Database, email_client: EmailClient):
        self.db = db_conn
        self.email = email_client

    def process(self, data: List[dict]) -> ProcessingResult:
        """Process and store data with notifications."""
        validated = self._validate_data(data)
        result = self.db.save(validated)
        self._notify_completion(result)
        return result
```

**Design Improvement Areas:**

If there's a **cleaner**, **more scalable**, or **simpler** design, highlight it and suggest improvements that would:

- Reduce code duplication through shared utilities
- Make unit testing easier
- Improve separation of concerns (single responsibility)
- Make unit testing easier through dependency injection
- Add clarity without adding complexity
- Prefer dataclasses for structured data

## Development Tools & Commands

### Package Management

```bash
# Add package
uv add package-name

# Sync project dependencies
uv sync
uv lock
```

### Testing

```bash
# Run unit tests (no network)
make test

# Don't run integration tests, as API keys must be set

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type checking
uv run --group lint mypy .
```

### Dependency Management Patterns

**Local Development Dependencies:**

```toml
[tool.uv.sources]
langchain-core = { path = "../core", editable = true }
langchain-tests = { path = "../standard-tests", editable = true }
```

**For tools, use the `@tool` decorator from `langchain_core.tools`:**

```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for relevant information.

    Args:
        query: The search query string.
    """
    # Implementation here
    return results
```

## Commit Standards

**Use Conventional Commits format for PR titles:**

- `feat(core): add multi-tenant support`
- `fix(cli): resolve flag parsing error`
- `docs: update API usage examples`
- `docs(openai): update API usage examples`

## Framework-Specific Guidelines

- Follow the existing patterns in `langchain-core` for base abstractions
- Use `langchain_core.callbacks` for execution tracking
- Implement proper streaming support where applicable
- Avoid deprecated components like legacy `LLMChain`

### Partner Integrations

- Follow the established patterns in existing partner libraries
- Implement standard interfaces (`BaseChatModel`, `BaseEmbeddings`, etc.)
- Include comprehensive integration tests
- Document API key requirements and authentication

---

## Quick Reference Checklist

Before submitting code changes:

- [ ] **Breaking Changes**: Verified no public API changes
- [ ] **Type Hints**: All functions have complete type annotations
- [ ] **Tests**: New functionality is fully tested
- [ ] **Security**: No dangerous patterns (eval, silent failures, etc.)
- [ ] **Documentation**: Google-style docstrings for public functions
- [ ] **Code Quality**: `make lint` and `make format` pass
- [ ] **Architecture**: Suggested improvements where applicable
- [ ] **Commit Message**: Follows Conventional Commits format

# Google-specific instructions

You can find the official SDK documentation and code samples here:
<https://ai.google.dev/gemini-api/docs>

## Golden Rule: Use the Correct and Current SDK

- **Library Name:** Google GenAI SDK
- **Python Package:** `google-genai`
- **Legacy Libraries**: (`google-generativeai` and `google-ai-generativelanguage`) are deprecated.

**APIs and Usage:**

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

## Basic Inference (Text Generation)

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

## Additional Capabilities and Configurations

Below are examples of advanced configurations.

### Thinking

Gemini 2.5 series models and above support thinking, which is on by default for
`gemini-2.5-flash`. It can be adjusted by using `thinking_budget` setting.
Setting it to zero turns thinking off, and will reduce latency.

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

- Minimum thinking budget for `gemini-2.5-pro` is `128` and thinking can not
    be turned off for that model.
- No models (apart from Gemini 2.5 series) support thinking or thinking
    budgets APIs. Do not try to adjust thinking budgets other models (such as
    `gemini-2.0-flash` or `gemini-2.0-pro`) otherwise it will cause syntax
    errors.

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

You can also set `temperature` or `max_output_tokens` within
`types.GenerateContentConfig`
**Avoid** setting `max_output_tokens`, `topP`, `topK` unless explicitly
requested by the user.

### Safety configurations

Avoid setting safety configurations unless explicitly requested by the user. If
explicitly asked for by the user, here is a sample API:

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

For multi-turn conversations, use the `chats` service to maintain conversation
history.

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

Use structured outputs to force the model to return a response that conforms to
a specific Pydantic schema.

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

#### Function Calling (Tools)

You can provide the model with tools (functions) it can use to bring in external
information to answer a question or act on a request outside the model.

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

### Generate Images

Here's how to generate images using the Imagen models. Start with the fast model
as it should cover most use-cases, and move to the more standard or the ultra
models for advanced use-cases.

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

Editing images is better done using the Gemini native image generation model,
and it is recommended to use chat mode. Configs are not supported in this model
(except modality).

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

### Generate Videos

Here's how to generate videos using the Veo models. Usage of Veo can be costly,
so after generating code for it, give user a heads up to check pricing for Veo.
Start with the fast model since the result quality is usually sufficient, and
swap to the larger model if needed.

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

### Search Grounding

Google Search can be used as a tool for grounding queries that with up to date
information from the web.

**Correct**

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

The output `response.text` will likely not be in JSON format, do not attempt to
parse it as JSON.

### Content and Part Hierarchy

While the simpler API call is often sufficient, you may run into scenarios where
you need to work directly with the underlying `Content` and `Part` objects for
more explicit control. These are the fundamental building blocks of the
`generate_content` API.

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

The list of APIs and capabilities above are not comprehensive. If users ask you
to generate code for a capability not provided above, refer them to
ai.google.dev/gemini-api/docs.

## Useful Links

- Documentation: ai.google.dev/gemini-api/docs
- API Keys and Authentication: ai.google.dev/gemini-api/docs/api-key
- Models: ai.google.dev/models
- API Pricing: ai.google.dev/pricing
- Rate Limits: ai.google.dev/rate-limits
