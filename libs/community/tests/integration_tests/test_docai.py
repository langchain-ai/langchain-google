"""Integration tests for the Google Cloud DocAI parser."""

import os

import pytest
from langchain_core.document_loaders.blob_loaders import Blob

from langchain_google_community.docai import DocAIParser

@pytest.mark.extended
@pytest.mark.parametrize("processor_name", [os.environ["PROCESSOR_NAME"]])
def test_docai_layout_parser(processor_name: str) -> None:
    parser = DocAIParser(processor_name=processor_name, location="us")
    assert parser._use_layout_parser is True
    blob = Blob(
        path="gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf"
    )
    docs = list(parser.online_process(blob=blob))
    assert len(docs) == 11
