"""Tests for the Google Cloud DocAI parser."""

import os
from unittest.mock import ANY, patch

import pytest

from langchain_google_community.docai import DocAIParser
from langchain_core.document_loaders.blob_loaders import Blob

def test_docai_parser_valid_processor_name() -> None:
    processor_name = "projects/123456/locations/us-central1/processors/ab123dfg"
    with patch("google.cloud.documentai.DocumentProcessorServiceClient") as test_client:
        parser = DocAIParser(processor_name=processor_name, location="us")
        test_client.assert_called_once_with(client_options=ANY, client_info=ANY)
        assert parser._processor_name == processor_name


@pytest.mark.parametrize(
    "processor_name",
    ["projects/123456/locations/us-central1/processors/ab123dfg:publish", "ab123dfg"],
)
def test_docai_parser_invalid_processor_name(processor_name: str) -> None:
    with patch("google.cloud.documentai.DocumentProcessorServiceClient"):
        with pytest.raises(ValueError):
            _ = DocAIParser(processor_name=processor_name, location="us")

@pytest.mark.parametrize(
    "processor_name",
    [os.environ["PROCESSOR_NAME"]]
)
def test_docai_layout_parser(processor_name: str) -> None:
    parser = DocAIParser(processor_name=processor_name, location="us")
    assert parser._use_layout_parser == True
    blob = Blob(path="gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf")
    docs = list(parser.online_process(blob=blob))
    assert len(docs) == 11