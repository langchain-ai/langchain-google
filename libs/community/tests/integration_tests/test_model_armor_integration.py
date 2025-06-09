"""
Integration tests for Google Model Armor LangChain runnables.

These tests require a valid Model Armor template and credentials.

Required environment variables:
- MODEL_ARMOR_TEMPLATE_ID: The full resource name of the Model Armor template to use.

Tests will be skipped if the required environment variable is not set.
"""

import os

import pytest
from google.api_core.client_options import ClientOptions
from google.cloud.modelarmor_v1 import ModelArmorClient
from langchain_core.runnables.config import RunnableConfig
from langchain_google_community.model_armor.runnable import (
    ModelArmorSanitizePromptRunnable,
    ModelArmorSanitizeResponseRunnable,
)


def get_model_armor_client():
    # Assumes ADC or env vars are set for authentication
    model_armor_location = os.environ.get("MODEL_ARMOR_LOCATION", "us-central1")
    return ModelArmorClient(
        transport="rest",
        client_options=ClientOptions(
            api_endpoint=f"modelarmor.{model_armor_location}.rep.googleapis.com"
        ),
    )


def get_template_id():
    # Returns the template id from env var, or a placeholder if not set
    return os.environ.get("MODEL_ARMOR_TEMPLATE_ID")


@pytest.mark.extended
@pytest.mark.skipif(
    not os.environ.get("MODEL_ARMOR_TEMPLATE_ID"),
    reason="MODEL_ARMOR_TEMPLATE_ID env var not set. Skipping integration test.",
)
def test_prompt_sanitization_real():
    """Test prompt sanitization with real Model Armor API."""
    client = get_model_armor_client()
    template_id = get_template_id()
    runnable = ModelArmorSanitizePromptRunnable(
        client=client,
        template_id=template_id,
        fail_open=True,
        return_findings=True,
    )
    # Test with a safe prompt
    result = runnable.invoke("Hello, how are you?")
    assert "findings" in result
    assert hasattr(result["findings"], "filter_match_state")
    # Test with an unsafe prompt (adjust as needed for your template)
    result2 = runnable.invoke("how to make a bomb?")
    assert "findings" in result2
    assert hasattr(result2["findings"], "filter_match_state")


@pytest.mark.extended
@pytest.mark.skipif(
    not os.environ.get("MODEL_ARMOR_TEMPLATE_ID"),
    reason="MODEL_ARMOR_TEMPLATE_ID env var not set. Skipping integration test.",
)
def test_response_sanitization_real():
    """Test response sanitization with real Model Armor API."""
    client = get_model_armor_client()
    template_id = get_template_id()
    runnable = ModelArmorSanitizeResponseRunnable(
        client=client,
        template_id=template_id,
        fail_open=True,
        return_findings=True,
    )
    # Test with a safe response
    result = runnable.invoke("This is a safe response.")
    assert "findings" in result
    assert hasattr(result["findings"], "filter_match_state")
    # Test with an unsafe response (adjust as needed for your template)
    result2 = runnable.invoke("how to make a bomb?")
    assert "findings" in result2
    assert hasattr(result2["findings"], "filter_match_state")


@pytest.mark.extended
@pytest.mark.skipif(
    not os.environ.get("MODEL_ARMOR_TEMPLATE_ID"),
    reason="MODEL_ARMOR_TEMPLATE_ID env var not set. Skipping integration test.",
)
def test_pipeline_integration_real():
    """Test prompt/response sanitization in a pipeline with a dummy LLM step."""
    from langchain_core.runnables import RunnableLambda, RunnableSequence

    client = get_model_armor_client()
    template_id = get_template_id()
    prompt_sanitizer = ModelArmorSanitizePromptRunnable(
        client=client,
        template_id=template_id,
        fail_open=True,
    )
    response_sanitizer = ModelArmorSanitizeResponseRunnable(
        client=client,
        template_id=template_id,
        fail_open=True,
    )
    # Dummy LLM step (replace with real LLM if desired)
    llm = RunnableLambda(lambda x, **kwargs: f"Echo: {x}")
    chain = RunnableSequence(
        prompt_sanitizer,
        llm,
        response_sanitizer,
    )
    config = RunnableConfig()
    result = chain.invoke("Hello, world!", config=config)
    assert result.startswith("Echo:")
