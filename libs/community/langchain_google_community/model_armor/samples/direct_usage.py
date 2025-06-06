"""
Sample to create and invoke Model Armor Runnables independently.
"""

from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor_v1 import ModelArmorClient

from langchain_google_community.model_armor import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

# TODO (Developer): Replace with your own project, location, and template IDs.
project_id = "my-project"
location_id = "us-central1"
template_id = "my-template"


# Refer here for more details on Model Armor Client Options:
# https://cloud.google.com/python/docs/reference/google-cloud-modelarmor/latest/google.cloud.modelarmor_v1.services.model_armor.ModelArmorClient

gcp_client = ModelArmorClient(
    client_options=ClientOptions(
        api_endpoint=f"modelarmor.{location_id}.rep.googleapis.com"
    )
)
template_id = f"projects/{project_id}/locations/{location_id}/templates/{template_id}"

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
