"""
Example usage of Model Armor runnables with LangChain's callback event handler
in chain.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor import ModelArmorClient
from langchain_community.llms.vertexai import (
    VertexAI,
)

# Replace with your preferred LLM
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableSequence

from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

# TODO (Developer): Replace with your own project, location, and template IDs.
project_id = "ma-crest-data-test"
location_id = "us-central1"
template_id = "temp-mihir-basic-sdp"

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
    client, template_name, fail_open=True
)
response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client, template_name, fail_open=True
)


# Callback handler to listen to custom event and trigger/create alert
class AlertingCallbackHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Handle custom events triggered by the runnables in the chain.

        In this example, it will print an alert message if a Model Armor finding
        is detected.

        :param name: The name of the custom event.
        :param data: The data associated with the custom event.
        :param run_id: The unique identifier of the run.
        :param tags: The tags associated with the run.
        :param metadata: The metadata associated with the run.
        :param kwargs: Additional keyword arguments.
        """
        # Check if custom event is for Model Armor findings
        if name == "on_model_armor_finding":
            # Update your alerting logic as per requirements
            print(
                "[ALERT] Unsafe content flagged "
                f"by Model Armor with findings: {data}."
            )


# Create the full processing chain
chain: Runnable = RunnableSequence(prompt_sanitizer | llm | response_sanitizer)


# Invoke the chain with user input
try:
    user_prompt = "Tell me something dangerous."
    result = chain.invoke(user_prompt, {"callbacks": [AlertingCallbackHandler()]})
    print("Final Output:", result)
except ValueError as e:
    print("Chain execution stopped due to blocked content:", str(e))
