from typing import Any
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

def test_validate_response_modalities() -> None:
    """Test response modality validation logic directly."""
    # 1. Setup a model with a known profile
    llm = ChatGoogleGenerativeAI(model="gemini-fake", api_key="test_key")
    
    # Manually inject a profile to ensure deterministic testing
    llm.profile = {"output_modalities": ["TEXT", "AUDIO"]}

    # 2. Test VALID configuration (should pass)
    try:
        valid_config = {"response_modalities": ["TEXT"]}
        llm._validate_response_modalities(valid_config)
    except ValueError:
        pytest.fail("Validation raised ValueError for supported modality 'TEXT'")

    # 3. Test INVALID configuration (should fail)
    invalid_config = {"response_modalities": ["IMAGE"]}
    with pytest.raises(ValueError) as exc_info:
        llm._validate_response_modalities(invalid_config)
    
    # Check error message content
    error_msg = str(exc_info.value)
    assert "does not support output modality 'IMAGE'" in error_msg
    assert "Supported modalities are: ['TEXT', 'AUDIO']" in error_msg

def test_validate_no_modalities_set() -> None:
    """Test that nothing breaks if no modalities are requested."""
    llm = ChatGoogleGenerativeAI(model="gemini-fake", api_key="test_key")
    llm.profile = {"output_modalities": ["TEXT"]}
    
    # Should not raise error
    llm._validate_response_modalities({})
