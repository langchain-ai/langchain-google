import os
from typing import List

import pytest
from google.cloud import discoveryengine_v1alpha  # type: ignore
from langchain_core.documents import Document

from langchain_google_community.vertex_check_grounding import (
    VertexAICheckGroundingWrapper,
)


@pytest.fixture
def input_documents() -> List[Document]:
    return [
        Document(
            page_content=(
                "Born in the German Empire, Einstein moved to Switzerland in 1895, "
                "forsaking his German citizenship (as a subject of the Kingdom of "
                "Württemberg)[note 1] the following year. In 1897, at the age of "
                "seventeen, he enrolled in the mathematics and physics teaching "
                "diploma program at the Swiss federal polytechnic school in Zürich, "
                "graduating in 1900. In 1901, he acquired Swiss citizenship, which "
                "he kept for the rest of his life. In 1903, he secured a permanent "
                "position at the Swiss Patent Office in Bern. In 1905, he submitted "
                "a successful PhD dissertation to the University of Zurich. In 1914, "
                "he moved to Berlin in order to join the Prussian Academy of Sciences "
                "and the Humboldt University of Berlin. In 1917, he became director "
                "of the Kaiser Wilhelm Institute for Physics; he also became a German "
                "citizen again, this time as a subject of the Kingdom of Prussia."
                "\nIn 1933, while he was visiting the United States, Adolf Hitler came "
                'to power in Germany. Horrified by the Nazi "war of extermination" '
                "against his fellow Jews,[12] Einstein decided to remain in the US, "
                "and was granted American citizenship in 1940.[13] On the eve of World "
                "War II, he endorsed a letter to President Franklin D. Roosevelt "
                "alerting him to the potential German nuclear weapons program and "
                "recommending that the US begin similar research. Einstein supported "
                "the Allies but generally viewed the idea of nuclear weapons with "
                "great dismay.[14]"
            ),
            metadata={
                "language": "en",
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                "title": "Albert Einstein - Wikipedia",
            },
        ),
        Document(
            page_content=(
                "Life and career\n"
                "Childhood, youth and education\n"
                "See also: Einstein family\n"
                "Einstein in 1882, age\xa03\n"
                "Albert Einstein was born in Ulm,[19] in the Kingdom of Württemberg "
                "in the German Empire, on 14 March 1879.[20][21] His parents, secular "
                "Ashkenazi Jews, were Hermann Einstein, a salesman and engineer, and "
                "Pauline Koch. In 1880, the family moved to Munich's borough of "
                "Ludwigsvorstadt-Isarvorstadt, where Einstein's father and his uncle "
                "Jakob founded Elektrotechnische Fabrik J. Einstein & Cie, a company "
                "that manufactured electrical equipment based on direct current.[19]\n"
                "Albert attended a Catholic elementary school in Munich from the age "
                "of five. When he was eight, he was transferred to the Luitpold "
                "Gymnasium, where he received advanced primary and then secondary "
                "school education.[22]"
            ),
            metadata={
                "language": "en",
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                "title": "Albert Einstein - Wikipedia",
            },
        ),
    ]


@pytest.fixture
def grounded_generation_service_client() -> (
    discoveryengine_v1alpha.GroundedGenerationServiceClient
):
    return discoveryengine_v1alpha.GroundedGenerationServiceClient()


@pytest.fixture
def output_parser(
    grounded_generation_service_client: (
        discoveryengine_v1alpha.GroundedGenerationServiceClient
    ),
) -> VertexAICheckGroundingWrapper:
    return VertexAICheckGroundingWrapper(
        project_id=os.environ["PROJECT_ID"],
        location_id=os.environ.get("REGION", "global"),
        grounding_config=os.environ.get("GROUNDING_CONFIG", "default_grounding_config"),
        client=grounded_generation_service_client,
    )


@pytest.mark.extended
def test_integration_parse(
    output_parser: VertexAICheckGroundingWrapper,
    input_documents: List[Document],
) -> None:
    answer_candidate = "Ulm, in the Kingdom of Württemberg in the German Empire"
    response = output_parser.with_config(
        configurable={"documents": input_documents}
    ).invoke(answer_candidate)

    assert isinstance(response, VertexAICheckGroundingWrapper.CheckGroundingResponse)
    assert response.support_score >= 0 and response.support_score <= 1
    assert len(response.cited_chunks) > 0
    for chunk in response.cited_chunks:
        assert isinstance(chunk["chunk_text"], str)
        assert isinstance(chunk["source"], Document)
    assert len(response.claims) > 0
    for claim in response.claims:
        assert isinstance(claim["start_pos"], int)
        assert isinstance(claim["end_pos"], int)
        assert isinstance(claim["claim_text"], str)
        assert isinstance(claim["citation_indices"], list)
    assert isinstance(response.answer_with_citations, str)
