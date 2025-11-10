"""Test ChatGoogleVertexAI chat model."""

import os

import pytest

from langchain_google_vertexai._image_utils import image_bytes_to_b64_string
from langchain_google_vertexai._utils import load_image_from_gcs
from langchain_google_vertexai.model_garden import ChatAnthropicVertex


@pytest.mark.extended
def test_pdf_gcs_uri() -> None:
    gcs_uri = "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
    llm = ChatAnthropicVertex(
        model="claude-sonnet-4-5@20250929",
        location="us-east5",
        temperature=0.8,
        project=os.environ["PROJECT_ID"],
    )

    res = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    "Parse this pdf.",
                    {"type": "image_url", "image_url": {"url": gcs_uri}},
                ],
            }
        ]
    )
    assert len(res.content) > 100


@pytest.mark.extended
def test_pdf_byts() -> None:
    gcs_uri = "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
    llm = ChatAnthropicVertex(
        model="claude-sonnet-4-5@20250929",
        location="us-east5",
        temperature=0.8,
        project=os.environ["PROJECT_ID"],
    )
    image = load_image_from_gcs(gcs_uri, "kuligin-sandbox1")
    image_data = image_bytes_to_b64_string(image.data, "ascii", "pdf")

    res = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    "Parse this pdf.",
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ]
    )
    assert len(res.content) > 100


@pytest.mark.extended
def test_https_image() -> None:
    uri = "https://picsum.photos/seed/picsum/200/300.jpg"

    llm = ChatAnthropicVertex(
        model="claude-sonnet-4-5@20250929",
        location="us-east5",
        temperature=0.8,
        project=os.environ["PROJECT_ID"],
    )

    res = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    "Parse this pdf.",
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ]
    )
    assert len(res.content) > 10
