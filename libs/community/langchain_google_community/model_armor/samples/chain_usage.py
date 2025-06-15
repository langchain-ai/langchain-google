"""
Example usage of Model Armor runnables in a chain.
"""

from google.cloud.modelarmor import ModelArmorClient
from google.api_core.client_options import ClientOptions
from langchain_community.llms.vertexai import VertexAI  # Or any other LLM
from langchain_core.runnables import RunnableSequence
from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

# TODO (Developer): Replace with your own project, location, and template IDs.
project_id = "my-project"
location_id = "us-central1"
template_id = "my-template"

# Initialize Model Armor client.
client = ModelArmorClient(
    client_options=ClientOptions(
        api_endpoint=f"modelarmor.{location_id}.rep.googleapis.com"
    ),
)

# Define Model Armor template name.
# Ref: https://cloud.google.com/security-command-center/docs/manage-model-armor-templates
template_name = (
    f"projects/{project_id}/locations/{location_id}/templates/{template_id}"
)

# Initialize your LLM.
llm = VertexAI(model_name="gemini-pro")

# Initialize runnables.
prompt_sanitizer = ModelArmorSanitizePromptRunnable(
    client, template_name, fail_open=False
)
response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client, template_name, fail_open=False
)

# Define the chain sequence.
chain = RunnableSequence(prompt_sanitizer | llm | response_sanitizer)

# Invoke chain with user prompt.
try:
    user_prompt = "Tell me something dangerous."
    result = chain.invoke(user_prompt)
    print("Final Output:", result)

# If Model Armor detects any findings that match the configured filters, an exception will be thrown and subsequent chain execution will be halted.
except ValueError as e:
    print("Chain stopped:", e)
