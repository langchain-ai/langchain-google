import os

import pytest
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from langchain_google_community import CloudVisionLoader


@pytest.mark.skip(reason="CI/CD not ready.")
def test_parse_image() -> None:
    gcs_path = os.environ["IMAGE_GCS_PATH"]
    project = os.environ["PROJECT"]
    blob = Blob(path=gcs_path, data="")  # type: ignore
    loader = CloudVisionLoader(project=project)
    documents = loader.parse(blob)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert len(documents[0].page_content) > 1
