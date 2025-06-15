"""
Example usage of Model Armor runnables with custom runnables in chain.
"""

from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor import ModelArmorClient
from langchain_community.llms.vertexai import VertexAI  # Or any other LLM
from langchain_core.runnables import Runnable, RunnableSequence
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

# Initialize Model Armor runnables.
prompt_sanitizer = ModelArmorSanitizePromptRunnable(
    client, template_name, fail_open=True
)
response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client, template_name, fail_open=True
)

######### Custom runnable #########


# Define your custom runnable here. In this example, the following runnable replaces text in prompt_sanitizer runnable output.
class ReplaceTextRunnable(Runnable):
    def __init__(self, old_text: str, new_text: str):
        self.old_text = old_text
        self.new_text = new_text

    def run(self, input: str) -> str:
        return input.replace(self.old_text, self.new_text)


prompt_replacer = ReplaceTextRunnable("dangerous", "safe")

######### Custom runnable #########

# Define the chain sequence.
chain = RunnableSequence(
    prompt_sanitizer | prompt_replacer | llm | response_sanitizer
)

# TODO: add SDP accessibility message

# Invoke chain with user prompt.
user_prompt = "Tell me something dangerous."
result = chain.invoke(user_prompt)
print("Final Output:", result)
