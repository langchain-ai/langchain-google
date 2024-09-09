import os
from typing import List
from unittest.mock import create_autospec

import pytest
from google.cloud import discoveryengine_v1alpha  # type: ignore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from langchain_google_community.vertex_rank import VertexAIRank


class CustomRankingRetriever(BaseRetriever):
    """Retriever that directly uses a mock retriever and a ranking API."""

    base_retriever: BaseRetriever = Field(default=None)
    ranker: VertexAIRank = Field(default=None)

    def __init__(self, base_retriever: BaseRetriever, ranker: VertexAIRank):
        super().__init__()  # Call to the superclass's constructor
        self.base_retriever = base_retriever
        self.ranker = ranker

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve and rank documents according to the query."""
        # Retrieve documents using the base retriever
        documents = self.base_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )
        # Rank documents using VertexAIRank
        ranked_documents = list(self.ranker.compress_documents(documents, query))
        return ranked_documents


class MockVectorStoreRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [
            Document(
                page_content=(
                    "Life and career\nChildhood, youth and education\n"
                    "See also: Einstein family\nEinstein in 1882, age\xa03\n"
                    "Albert Einstein was born in Ulm,[19] in the Kingdom of "
                    "Württemberg in the German Empire, on 14 March 1879.[20][21] "
                    "His parents, secular Ashkenazi Jews, were Hermann Einstein, "
                    "a salesman and engineer, and Pauline Koch. In 1880, the "
                    "family moved to Munich's borough of Ludwigsvorstadt-"
                    "Isarvorstadt, where Einstein's father and his uncle Jakob "
                    "founded Elektrotechnische Fabrik J. Einstein & Cie, a "
                    "company that manufactured electrical equipment based on "
                    "direct current.[19]\nAlbert attended a Catholic elementary "
                    "school in Munich from the age of five. When he was eight, "
                    "he was transferred to the Luitpold Gymnasium, where he "
                    "received advanced primary and then secondary school "
                    "education.[22]"
                ),
                metadata={
                    "language": "en",
                    "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                    "title": "Albert Einstein - Wikipedia",
                },
            ),
            Document(
                page_content=(
                    "A volume of Einstein's letters released by Hebrew "
                    "University of Jerusalem in 2006[61] added further names to "
                    "the catalog of women with whom he was romantically "
                    "involved. They included Margarete Lebach (a married "
                    "Austrian),[62] Estella Katzenellenbogen (the rich owner of "
                    "a florist business), Toni Mendel (a wealthy Jewish widow) "
                    "and Ethel Michanowski (a Berlin socialite), with whom he "
                    "spent time and from whom he accepted gifts while married to "
                    "Löwenthal.[63][64] After being widowed, Einstein was "
                    "briefly in a relationship with Margarita Konenkova, thought "
                    "by some to be a Russian spy; her husband, the Russian "
                    "sculptor Sergei Konenkov, created the bronze bust of "
                    "Einstein at the Institute for Advanced Study at "
                    "Princeton.[65][66][failed verification]\n"
                    "Following an episode of acute mental illness at about the "
                    "age of twenty, Einstein's son Eduard was diagnosed with "
                    "schizophrenia.[67] He spent the remainder of his life "
                    "either in the care of his mother or in temporary "
                    "confinement in an asylum. After her death, he was committed "
                    "permanently to Burghölzli, the Psychiatric University "
                    "Hospital in Zürich.[68]"
                ),
                metadata={
                    "language": "en",
                    "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                    "title": "Albert Einstein - Wikipedia",
                },
            ),
            Document(
                page_content=(
                    "Marriages, relationships and children\n"
                    "Albert Einstein and Mileva Marić Einstein, 1912\n"
                    "Albert Einstein and Elsa Einstein, 1930\n"
                    "Correspondence between Einstein and Marić, discovered and "
                    "published in 1987, revealed that in early 1902, while "
                    "Marić was visiting her parents in Novi Sad, she gave birth "
                    "to a daughter, Lieserl. When Marić returned to Switzerland "
                    "it was without the child, whose fate is uncertain. A "
                    "letter of Einstein's that he wrote in September 1903 "
                    "suggests that the girl was either given up for adoption or "
                    "died of scarlet fever in infancy.[45][46]\n"
                    "Einstein and Marić married in January 1903. In May 1904, "
                    "their son Hans Albert was born in Bern, Switzerland. Their "
                    "son Eduard was born in Zürich in July 1910. In letters "
                    "that Einstein wrote to Marie Winteler in the months before "
                    "Eduard's arrival, he described his love for his wife as "
                    '"misguided" and mourned the "missed life" that he imagined '
                    "he would have enjoyed if he had married Winteler instead: "
                    '"I think of you in heartfelt love every spare minute and '
                    'am so unhappy as only a man can be."[47]'
                ),
                metadata={
                    "language": "en",
                    "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
                    "title": "Albert Einstein - Wikipedia",
                },
            ),
        ]


@pytest.fixture
def mock_vector_store_retriever() -> MockVectorStoreRetriever:
    mock_store = create_autospec(VectorStore, instance=True)
    return MockVectorStoreRetriever(vectorstore=mock_store)


@pytest.fixture
def rank_service_client() -> discoveryengine_v1alpha.RankServiceClient:
    return (
        discoveryengine_v1alpha.RankServiceClient()
    )  # Ensure you have credentials configured


@pytest.fixture
def ranker(
    rank_service_client: discoveryengine_v1alpha.RankServiceClient,
) -> VertexAIRank:
    return VertexAIRank(
        project_id=os.environ["PROJECT_ID"],
        location_id=os.environ.get("REGION", "global"),
        ranking_config=os.environ.get("RANKING_CONFIG", "default_ranking_config"),
        title_field="source",
        client=rank_service_client,
    )


@pytest.mark.extended
def test_compression_retriever(
    mock_vector_store_retriever: MockVectorStoreRetriever, ranker: VertexAIRank
) -> None:
    compression_retriever = CustomRankingRetriever(
        base_retriever=mock_vector_store_retriever, ranker=ranker
    )
    query = "What was the name of einstein's mother ?"
    compressed_docs = compression_retriever.get_relevant_documents(query)

    expected_docs = [
        Document(
            page_content=(
                "Life and career\nChildhood, youth and education\n"
                "See also: Einstein family\nEinstein in 1882, age\xa03\n"
                "Albert Einstein was born in Ulm,[19] in the Kingdom of "
                "Württemberg in the German Empire, on 14 March 1879.[20][21] "
                "His parents, secular Ashkenazi Jews, were Hermann Einstein, "
                "a salesman and engineer, and Pauline Koch. In 1880, the "
                "family moved to Munich's borough of Ludwigsvorstadt-"
                "Isarvorstadt, where Einstein's father and his uncle Jakob "
                "founded Elektrotechnische Fabrik J. Einstein & Cie, a "
                "company that manufactured electrical equipment based on "
                "direct current.[19]\nAlbert attended a Catholic elementary "
                "school in Munich from the age of five. When he was eight, "
                "he was transferred to the Luitpold Gymnasium, where he "
                "received advanced primary and then secondary school "
                "education.[22]"
            ),
            metadata={
                "id": "0",
                "relevance_score": 0.7599999904632568,
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
            },
        ),
        Document(
            page_content=(
                "Marriages, relationships and children\n"
                "Albert Einstein and Mileva Marić Einstein, 1912\n"
                "Albert Einstein and Elsa Einstein, 1930\n"
                "Correspondence between Einstein and Marić, discovered and "
                "published in 1987, revealed that in early 1902, while "
                "Marić was visiting her parents in Novi Sad, she gave birth "
                "to a daughter, Lieserl. When Marić returned to Switzerland "
                "it was without the child, whose fate is uncertain. A "
                "letter of Einstein's that he wrote in September 1903 "
                "suggests that the girl was either given up for adoption or "
                "died of scarlet fever in infancy.[45][46]\n"
                "Einstein and Marić married in January 1903. In May 1904, "
                "their son Hans Albert was born in Bern, Switzerland. Their "
                "son Eduard was born in Zürich in July 1910. In letters "
                "that Einstein wrote to Marie Winteler in the months before "
                "Eduard's arrival, he described his love for his wife as "
                '"misguided" and mourned the "missed life" that he imagined '
                "he would have enjoyed if he had married Winteler instead: "
                '"I think of you in heartfelt love every spare minute and '
                'am so unhappy as only a man can be."[47]'
            ),
            metadata={
                "id": "2",
                "relevance_score": 0.6399999856948853,
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
            },
        ),
        Document(
            page_content=(
                "A volume of Einstein's letters released by Hebrew "
                "University of Jerusalem in 2006[61] added further names to "
                "the catalog of women with whom he was romantically "
                "involved. They included Margarete Lebach (a married "
                "Austrian),[62] Estella Katzenellenbogen (the rich owner of "
                "a florist business), Toni Mendel (a wealthy Jewish widow) "
                "and Ethel Michanowski (a Berlin socialite), with whom he "
                "spent time and from whom he accepted gifts while married to "
                "Löwenthal.[63][64] After being widowed, Einstein was "
                "briefly in a relationship with Margarita Konenkova, thought "
                "by some to be a Russian spy; her husband, the Russian "
                "sculptor Sergei Konenkov, created the bronze bust of "
                "Einstein at the Institute for Advanced Study at "
                "Princeton.[65][66][failed verification]\n"
                "Following an episode of acute mental illness at about the "
                "age of twenty, Einstein's son Eduard was diagnosed with "
                "schizophrenia.[67] He spent the remainder of his life "
                "either in the care of his mother or in temporary "
                "confinement in an asylum. After her death, he was committed "
                "permanently to Burghölzli, the Psychiatric University "
                "Hospital in Zürich.[68]"
            ),
            metadata={
                "id": "1",
                "relevance_score": 0.10999999940395355,
                "source": "https://en.wikipedia.org/wiki/Albert_Einstein",
            },
        ),
    ]

    assert len(compressed_docs) == len(expected_docs)
    for doc, expected in zip(compressed_docs, expected_docs):
        assert doc.page_content == expected.page_content
        assert doc.metadata["id"] == expected.metadata["id"]
        assert float(doc.metadata.get("relevance_score", 0)) > 0
