"""
Example usage of Model Armor runnables with custom runnables in chain.
"""

from typing import Any, Optional

from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor import ModelArmorClient
from langchain_community.llms.vertexai import VertexAI  # Or any other LLM
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSequence

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
template_name = f"projects/{project_id}/locations/{location_id}/templates/{template_id}"

# Initialize your LLM.
llm = VertexAI(model_name="gemini-pro")

# Initialize Model Armor runnables.
prompt_sanitizer = ModelArmorSanitizePromptRunnable(
    client, template_name, fail_open=True, return_findings=True
)
response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client, template_name, fail_open=True
)


class SanitizedTextRunnable(Runnable):
    """
    A custom runnable that return sanitized text received from Model Armor with
    SDP template or the original input.

    TODO (Developer):
    Update/Create runnable as per your requirements to use in chain.
    """

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        if isinstance(input, dict):
            if input.get("findings"):
                findings: Any = input.get("findings")
                if findings.filter_results.get("sdp"):
                    sanitized_text = findings.filter_results.get(
                        "sdp"
                    ).sdp_filter_result.deidentify_result.data.text
                    return sanitized_text or input.get("prompt", input)
            return input.get("prompt", input)
        return input


prompt_replacer = SanitizedTextRunnable()


# Define the chain sequence.
chain: Runnable = RunnableSequence(
    prompt_sanitizer | prompt_replacer | llm | response_sanitizer
)

# Invoke chain with user prompt.
user_prompt = "get my email mail@support.com and my phone number 123-456-7890"
result = chain.invoke(user_prompt)
print("Final Output:", result)
