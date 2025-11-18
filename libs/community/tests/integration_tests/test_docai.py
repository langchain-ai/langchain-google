"""Integration tests for the Google Cloud DocAI parser."""

import os

import pytest
from langchain_core.document_loaders.blob_loaders import Blob

from langchain_google_community.docai import DocAIParser


@pytest.mark.extended
@pytest.mark.xfail(reason="TEMPORARY until dependency issues are resolved.")
def test_docai_layout_parser() -> None:
    processor_name = os.environ["PROCESSOR_NAME"]
    parser = DocAIParser(processor_name=processor_name, location="us")
    assert parser._use_layout_parser is True
    blob = Blob(
        data=None,
        path="gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/2022Q1_alphabet_earnings_release.pdf",
    )
    docs = list(parser.online_process(blob=blob))
    assert len(docs) == 11
