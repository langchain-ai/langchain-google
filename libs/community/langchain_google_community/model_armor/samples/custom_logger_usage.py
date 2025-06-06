"""
Example of using Model Armor Runnables with a custom logger in chain.
"""

import logging

from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor import ModelArmorClient
from langchain_community.llms.vertexai import VertexAI  # Or any other LLM
from langchain_core.runnables import Runnable, RunnableSequence

from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)

######### Custom logger starts #########

# Create a logger.
logger = logging.getLogger("model_armor")
logger.setLevel(logging.INFO)

# Create a file handler which logs even debug messages.
fh = logging.FileHandler("model_armor.log")
fh.setLevel(logging.INFO)

# Create and set the formatter for the handler.
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# Add the Handler to the logger.
logger.addHandler(fh)

######### Custom logger ends #########

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
template_name = f"projects/{project_id}/locations/{location_id}/templates/{template_id}"

# Create LLM.
llm = VertexAI(model_name="gemini-pro")

# Initialize Model Armor runnables.
prompt_sanitizer = ModelArmorSanitizePromptRunnable(
    client, template_name, fail_open=True, logger=logger
)
response_sanitizer = ModelArmorSanitizeResponseRunnable(
    client, template_name, fail_open=True, logger=logger
)

# Define the runnable sequence chain.
chain: Runnable = RunnableSequence(prompt_sanitizer | llm | response_sanitizer)

# Invoke the defined chain with the user prompt.
# Since fail_open is set to True, chain execution will not be interrupted
# in case Model Armor detects any findings.
result = chain.invoke("Tell me something dangerous.")
