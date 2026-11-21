"""URL retrieval outcomes must survive response conversion and streaming."""

import pytest
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from langchain_core.outputs import ChatGenerationChunk

from langchain_google_genai.chat_models import _response_to_result


@pytest.mark.parametrize("stream", [False, True])
def test_url_context_metadata_preserved(stream: bool) -> None:
    metadata = {
        "url_metadata": [
            {
                "retrieved_url": "https://example.com/",
                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
            },
            {
                "retrieved_url": "https://example.com/missing",
                "url_retrieval_status": "URL_RETRIEVAL_STATUS_ERROR",
            },
        ]
    }
    response = GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Summary"}]},
                    "url_context_metadata": metadata,
                }
            ]
        }
    )

    generation = _response_to_result(response, stream=stream).generations[0]

    assert generation.message.response_metadata["url_context_metadata"] == metadata
    assert generation.generation_info is not None
    assert generation.generation_info["url_context_metadata"] == metadata


def test_url_context_metadata_absent() -> None:
    response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="Hello")]))]
    )

    generation = _response_to_result(response).generations[0]

    assert "url_context_metadata" not in generation.message.response_metadata
    assert "url_context_metadata" not in (generation.generation_info or {})


def test_url_context_metadata_in_final_stream_chunk() -> None:
    first = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="Summary")]))]
    )
    metadata = {
        "url_metadata": [
            {
                "retrieved_url": "https://example.com/",
                "url_retrieval_status": "URL_RETRIEVAL_STATUS_SUCCESS",
            }
        ]
    }
    final = GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "STOP",
                    "url_context_metadata": metadata,
                }
            ]
        }
    )
    first_chunk = _response_to_result(first, stream=True).generations[0]
    final_chunk = _response_to_result(final, stream=True).generations[0]
    assert isinstance(first_chunk, ChatGenerationChunk)
    assert isinstance(final_chunk, ChatGenerationChunk)

    merged = first_chunk + final_chunk

    assert merged.message.content == "Summary"
    assert merged.message.response_metadata["url_context_metadata"] == metadata
    assert merged.generation_info is not None
    assert merged.generation_info["url_context_metadata"] == metadata
