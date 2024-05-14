import os

import pytest
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from langchain_google_community import CloudVisionLoader, CloudVisionParser


@pytest.mark.extended
def test_parse_image() -> None:
    gcs_path = os.environ["IMAGE_GCS_PATH"]
    project = os.environ["PROJECT_ID"]
    blob = Blob(path=gcs_path, data="")  # type: ignore
    loader = CloudVisionParser(project=project)
    documents = loader.parse(blob)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert len(documents[0].page_content) > 1


@pytest.mark.extended
def test_load_image() -> None:
    gcs_path = os.environ["IMAGE_GCS_PATH"]
    project = os.environ["PROJECT_ID"]
    loader = CloudVisionLoader(project=project, file_path=gcs_path)
    documents = loader.load()
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert len(documents[0].page_content) > 1
