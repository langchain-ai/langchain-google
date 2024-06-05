from unittest.mock import Mock

import pytest
from google.cloud import discoveryengine_v1alpha  # type: ignore
from langchain_core.documents import Document

from langchain_google_community.vertex_check_grounding import (
    VertexAICheckGroundingWrapper,
)


@pytest.fixture
def mock_check_grounding_service_client() -> Mock:
    mock_client = Mock(spec=discoveryengine_v1alpha.GroundedGenerationServiceClient)
    mock_client.check_grounding.return_value = discoveryengine_v1alpha.CheckGroundingResponse(  # noqa: E501
        support_score=0.9919261932373047,
        cited_chunks=[
            discoveryengine_v1alpha.FactChunk(
                chunk_text=(
                    "Life and career\n"
                    "Childhood, youth and education\n"
                    "See also: Einstein family\n"
                    "Einstein in 1882, age\xa03\n"
                    "Albert Einstein was born in Ulm,[19] in the Kingdom of "
                    "Württemberg in the German Empire, on 14 March "
                    "1879.[20][21] His parents, secular Ashkenazi Jews, were "
                    "Hermann Einstein, a salesman and engineer, and "
                    "Pauline Koch. In 1880, the family moved to Munich's "
                    "borough of Ludwigsvorstadt-Isarvorstadt, where "
                    "Einstein's father and his uncle Jakob founded "
                    "Elektrotechnische Fabrik J. Einstein & Cie, a company "
                    "that manufactured electrical equipment based on direct "
                    "current.[19]\n"
                    "Albert attended a Catholic elementary school in Munich "
                    "from the age of five. When he was eight, he was "
                    "transferred to the Luitpold Gymnasium, where he received "
                    "advanced primary and then secondary school education.[22]"
                ),
                source="0",
            ),
        ],
        claims=[
            discoveryengine_v1alpha.CheckGroundingResponse.Claim(
                start_pos=0,
                end_pos=56,
                claim_text="Ulm, in the Kingdom of Württemberg in the German Empire",
                citation_indices=[0],
            ),
        ],
    )
    return mock_client


def test_parse(mock_check_grounding_service_client: Mock) -> None:
    output_parser = VertexAICheckGroundingWrapper(
        project_id="test-project",
        client=mock_check_grounding_service_client,
    )
    documents = [
        Document(
            page_content=(
                "Life and career\n"
                "Childhood, youth and education\n"
                "See also: Einstein family\n"
                "Einstein in 1882, age\xa03\n"
                "Albert Einstein was born in Ulm,[19] in the Kingdom of "
                "Württemberg in the German Empire, on 14 March "
                "1879.[20][21] His parents, secular Ashkenazi Jews, were "
                "Hermann Einstein, a salesman and engineer, and "
                "Pauline Koch. In 1880, the family moved to Munich's "
                "borough of Ludwigsvorstadt-Isarvorstadt, where "
                "Einstein's father and his uncle Jakob founded "
                "Elektrotechnische Fabrik J. Einstein & Cie, a company that "
                "manufactured electrical equipment based on direct current.[19]\n"
                "Albert attended a Catholic elementary school in Munich "
                "from the age of five. When he was eight, he was "
                "transferred to the Luitpold Gymnasium, where he received "
                "advanced primary and then secondary school education.[22]"
            ),
            metadata={
                "language": "en",
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                "title": "Albert Einstein - Wikipedia",
            },
        ),
    ]
    answer_candidate = "Ulm, in the Kingdom of Württemberg in the German Empire"
    response = output_parser.with_config(configurable={"documents": documents}).invoke(
        answer_candidate
    )

    assert response == VertexAICheckGroundingWrapper.CheckGroundingResponse(
        support_score=0.9919261932373047,
        cited_chunks=[
            {
                "chunk_text": (
                    "Life and career\n"
                    "Childhood, youth and education\n"
                    "See also: Einstein family\n"
                    "Einstein in 1882, age\xa03\n"
                    "Albert Einstein was born in Ulm,[19] in the Kingdom of "
                    "Württemberg in the German Empire, on 14 March "
                    "1879.[20][21] His parents, secular Ashkenazi Jews, were "
                    "Hermann Einstein, a salesman and engineer, and "
                    "Pauline Koch. In 1880, the family moved to Munich's "
                    "borough of Ludwigsvorstadt-Isarvorstadt, where "
                    "Einstein's father and his uncle Jakob founded "
                    "Elektrotechnische Fabrik J. Einstein & Cie, a company that "
                    "manufactured electrical equipment based on direct current.[19]\n"
                    "Albert attended a Catholic elementary school in Munich "
                    "from the age of five. When he was eight, he was "
                    "transferred to the Luitpold Gymnasium, where he received "
                    "advanced primary and then secondary school education.[22]"
                ),
                "source": documents[0],
            },
        ],
        claims=[
            {
                "start_pos": 0,
                "end_pos": 56,
                "claim_text": "Ulm, in the Kingdom of Württemberg in the German Empire",
                "citation_indices": [0],
            },
        ],
        answer_with_citations=(
            "Ulm, in the Kingdom of Württemberg in the German Empire[0]"
        ),
    )
