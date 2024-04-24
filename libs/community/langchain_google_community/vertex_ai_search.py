"""Retriever wrapper for Google Vertex AI Search.

Set the following environment variables before the tests:
export PROJECT_ID=... - set to your Google Cloud project ID
export DATA_STORE_ID=... - the ID of the search engine to use for the test
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InvalidArgument
from google.protobuf.json_format import MessageToDict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud.discoveryengine_v1beta import (  # type: ignore[import, attr-defined]
        ConversationalSearchServiceClient,
        SearchRequest,
        SearchResult,
        SearchServiceClient,
    )


class _BaseVertexAISearchRetriever(BaseModel):
    project_id: str
    """Google Cloud Project ID."""
    data_store_id: str
    """Vertex AI Search data store ID."""
    location_id: str = "global"
    """Vertex AI Search data store location."""
    serving_config_id: str = "default_config"
    """Vertex AI Search serving config ID."""
    credentials: Any = None
    """The default custom credentials (google.auth.credentials.Credentials) to use
    when making API calls. If not provided, credentials will be ascertained from
    the environment."""
    engine_data_type: int = Field(default=0, ge=0, le=2)
    """ Defines the Vertex AI Search data type
    0 - Unstructured data 
    1 - Structured data
    2 - Website data
    """

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates the environment."""
        try:
            from google.cloud import discoveryengine_v1beta  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please, install vertexaisearch dependency group: "
                "poetry install --with vertexaisearch"
            ) from exc

        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")

        try:
            # For backwards compatibility
            search_engine_id = get_from_dict_or_env(
                values, "search_engine_id", "SEARCH_ENGINE_ID"
            )

            if search_engine_id:
                warnings.warn(
                    "The `search_engine_id` parameter is deprecated. Use `data_store_id` instead.",  # noqa: E501
                    DeprecationWarning,
                )
                values["data_store_id"] = search_engine_id
        except:  # noqa: E722
            pass

        values["data_store_id"] = get_from_dict_or_env(
            values, "data_store_id", "DATA_STORE_ID"
        )

        return values

    @property
    def client_options(self) -> "ClientOptions":
        return ClientOptions(
            api_endpoint=(
                f"{self.location_id}-discoveryengine.googleapis.com"
                if self.location_id != "global"
                else None
            )
        )

    def _convert_structured_search_response(
        self, results: Sequence[SearchResult]
    ) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        documents: List[Document] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )

            documents.append(
                Document(
                    page_content=json.dumps(document_dict.get("struct_data", {})),
                    metadata={"id": document_dict["id"], "name": document_dict["name"]},
                )
            )

        return documents

    def _convert_unstructured_search_response(
        self, results: Sequence[SearchResult], chunk_type: str
    ) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        documents: List[Document] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            derived_struct_data = document_dict.get("derived_struct_data")
            if not derived_struct_data:
                continue

            doc_metadata = document_dict.get("struct_data", {})
            doc_metadata["id"] = document_dict["id"]

            if chunk_type not in derived_struct_data:
                continue

            for chunk in derived_struct_data[chunk_type]:
                doc_metadata["source"] = derived_struct_data.get("link", "")

                if chunk_type == "extractive_answers":
                    doc_metadata["source"] += f":{chunk.get('pageNumber', '')}"

                documents.append(
                    Document(
                        page_content=chunk.get("content", ""), metadata=doc_metadata
                    )
                )

        return documents

    def _convert_website_search_response(
        self, results: Sequence[SearchResult], chunk_type: str
    ) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        documents: List[Document] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            derived_struct_data = document_dict.get("derived_struct_data")
            if not derived_struct_data:
                continue

            doc_metadata = document_dict.get("struct_data", {})
            doc_metadata["id"] = document_dict["id"]
            doc_metadata["source"] = derived_struct_data.get("link", "")

            if chunk_type not in derived_struct_data:
                continue

            text_field = "snippet" if chunk_type == "snippets" else "content"

            for chunk in derived_struct_data[chunk_type]:
                documents.append(
                    Document(
                        page_content=chunk.get(text_field, ""), metadata=doc_metadata
                    )
                )

        if not documents:
            print(f"No {chunk_type} could be found.")  # noqa: T201
            if chunk_type == "extractive_answers":
                print(  # noqa: T201
                    "Make sure that your data store is using Advanced Website "
                    "Indexing.\n"
                    "https://cloud.google.com/generative-ai-app-builder/docs/about-advanced-features#advanced-website-indexing"  # noqa: E501
                )

        return documents


class VertexAISearchRetriever(BaseRetriever, _BaseVertexAISearchRetriever):
    """`Google Vertex AI Search` retriever.

    For a detailed explanation of the Vertex AI Search concepts
    and configuration parameters, refer to the product documentation.
    https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction
    """

    filter: Optional[str] = None
    """Filter expression."""
    get_extractive_answers: bool = False
    """If True return Extractive Answers, otherwise return Extractive Segments or Snippets."""  # noqa: E501
    max_documents: int = Field(default=5, ge=1, le=100)
    """The maximum number of documents to return."""
    max_extractive_answer_count: int = Field(default=1, ge=1, le=5)
    """The maximum number of extractive answers returned in each search result.
    At most 5 answers will be returned for each SearchResult.
    """
    max_extractive_segment_count: int = Field(default=1, ge=1, le=1)
    """The maximum number of extractive segments returned in each search result.
    Currently one segment will be returned for each SearchResult.
    """
    query_expansion_condition: int = Field(default=1, ge=0, le=2)
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified query expansion condition. In this case, server behavior defaults 
        to disabled
    1 - Disabled query expansion. Only the exact search query is used, even if 
        SearchResponse.total_size is zero.
    2 - Automatic query expansion built by the Search API.
    """
    spell_correction_mode: int = Field(default=2, ge=0, le=2)
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified spell correction mode. In this case, server behavior defaults 
        to auto.
    1 - Suggestion only. Search API will try to find a spell suggestion if there is any
        and put in the `SearchResponse.corrected_query`.
        The spell suggestion will not be used as the search query.
    2 - Automatic spell correction built by the Search API.
        Search will be based on the corrected query if found.
    """

    _client: SearchServiceClient
    _serving_config: str

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **kwargs: Any) -> None:
        """Initializes private fields."""
        try:
            from google.cloud.discoveryengine_v1beta import SearchServiceClient
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please, install vertexaisearch dependency group: "
                "`pip install langchain-google-community[vertexaisearch]`"
            ) from exc

        super().__init__(**kwargs)

        #  For more information, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
        self._client = SearchServiceClient(
            credentials=self.credentials,
            client_options=self.client_options,
            client_info=get_client_info(module="vertex-ai-search"),
        )

        self._serving_config = self._client.serving_config_path(
            project=self.project_id,
            location=self.location_id,
            data_store=self.data_store_id,
            serving_config=self.serving_config_id,
        )

    def _get_content_spec_kwargs(self) -> Optional[Dict[str, Any]]:
        """Prepares a ContentSpec object."""

        from google.cloud.discoveryengine_v1beta import SearchRequest

        if self.engine_data_type == 0:
            if self.get_extractive_answers:
                extractive_content_spec = (
                    SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                        max_extractive_answer_count=self.max_extractive_answer_count,
                    )
                )
            else:
                extractive_content_spec = (
                    SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                        max_extractive_segment_count=self.max_extractive_segment_count,
                    )
                )
            content_search_spec = dict(extractive_content_spec=extractive_content_spec)
        elif self.engine_data_type == 1:
            content_search_spec = None
        elif self.engine_data_type == 2:
            content_search_spec = dict(
                extractive_content_spec=SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=self.max_extractive_answer_count,
                ),
                snippet_spec=SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True
                ),
            )
        else:
            raise NotImplementedError(
                "Only data store type 0 (Unstructured), 1 (Structured),"
                "or 2 (Website) are supported currently."
                + f" Got {self.engine_data_type}"
            )
        return content_search_spec

    def _create_search_request(self, query: str) -> SearchRequest:
        """Prepares a SearchRequest object."""
        from google.cloud.discoveryengine_v1beta import SearchRequest

        query_expansion_spec = SearchRequest.QueryExpansionSpec(
            condition=self.query_expansion_condition,
        )

        spell_correction_spec = SearchRequest.SpellCorrectionSpec(
            mode=self.spell_correction_mode
        )

        content_search_spec_kwargs = self._get_content_spec_kwargs()

        if content_search_spec_kwargs is not None:
            content_search_spec = SearchRequest.ContentSearchSpec(
                **content_search_spec_kwargs
            )
        else:
            content_search_spec = None

        return SearchRequest(
            query=query,
            filter=self.filter,
            serving_config=self._serving_config,
            page_size=self.max_documents,
            content_search_spec=content_search_spec,
            query_expansion_spec=query_expansion_spec,
            spell_correction_spec=spell_correction_spec,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""

        search_request = self._create_search_request(query)

        try:
            response = self._client.search(search_request)
        except InvalidArgument as exc:
            raise type(exc)(
                exc.message
                + " This might be due to engine_data_type not set correctly."
            )

        if self.engine_data_type == 0:
            chunk_type = (
                "extractive_answers"
                if self.get_extractive_answers
                else "extractive_segments"
            )
            documents = self._convert_unstructured_search_response(
                response.results, chunk_type
            )
        elif self.engine_data_type == 1:
            documents = self._convert_structured_search_response(response.results)
        elif self.engine_data_type == 2:
            chunk_type = (
                "extractive_answers" if self.get_extractive_answers else "snippets"
            )
            documents = self._convert_website_search_response(
                response.results, chunk_type
            )
        else:
            raise NotImplementedError(
                "Only data store type 0 (Unstructured), 1 (Structured),"
                "or 2 (Website) are supported currently."
                + f" Got {self.engine_data_type}"
            )

        return documents


class VertexAIMultiTurnSearchRetriever(BaseRetriever, _BaseVertexAISearchRetriever):
    """`Google Vertex AI Search` retriever for multi-turn conversations."""

    conversation_id: str = "-"
    """Vertex AI Search Conversation ID."""

    _client: ConversationalSearchServiceClient
    _serving_config: str

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        from google.cloud.discoveryengine_v1beta import (
            ConversationalSearchServiceClient,
        )

        self._client = ConversationalSearchServiceClient(
            credentials=self.credentials,
            client_options=self.client_options,
            client_info=get_client_info(module="vertex-ai-search"),
        )

        self._serving_config = self._client.serving_config_path(
            project=self.project_id,
            location=self.location_id,
            data_store=self.data_store_id,
            serving_config=self.serving_config_id,
        )

        if self.engine_data_type == 1:
            raise NotImplementedError(
                "Data store type 1 (Structured)"
                "is not currently supported for multi-turn search."
                + f" Got {self.engine_data_type}"
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        from google.cloud.discoveryengine_v1beta import (
            ConverseConversationRequest,
            TextInput,
        )

        request = ConverseConversationRequest(
            name=self._client.conversation_path(
                self.project_id,
                self.location_id,
                self.data_store_id,
                self.conversation_id,
            ),
            serving_config=self._serving_config,
            query=TextInput(input=query),
        )
        response = self._client.converse_conversation(request)

        if self.engine_data_type == 2:
            return self._convert_website_search_response(
                response.search_results, "extractive_answers"
            )

        return self._convert_unstructured_search_response(
            response.search_results, "extractive_answers"
        )


class VertexAISearchSummaryTool(BaseTool, VertexAISearchRetriever):
    """Class that exposes a tool to interface with an App in Vertex Search and
    Conversation and get the summary of the documents retrieved.
    """

    summary_prompt: Optional[str] = None
    """Prompt for the summarization agent"""

    summary_result_count: int = 3
    """ Number of documents to include in the summary"""

    summary_include_citations: bool = True
    """ Whether to include citations in the summary """

    summary_spec_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Additional kwargs for `SearchRequest.ContentSearchSpec.SummarySpec`"""

    class Config(VertexAISearchRetriever.Config):
        """Redefinition to specify that inherits config from `VertexAISearchRetriever`
        not BaseTool
        """

    def _get_content_spec_kwargs(self) -> Optional[Dict[str, Any]]:
        """Adds additional summary_spec parameters to the configuration of the search.
        Returns:
            kwargs for the specification of the content.
        """
        from google.cloud.discoveryengine_v1beta import SearchRequest

        kwargs = super()._get_content_spec_kwargs() or {}

        kwargs["summary_spec"] = SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=self.summary_result_count,
            include_citations=self.summary_include_citations,
            model_prompt_spec=SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble=self.summary_prompt
            ),
            **self.summary_spec_kwargs,
        )

        return kwargs

    def _run(self, user_query: str) -> str:
        """Runs the tool.
        Args:
            search_query: The query to run by the agent.
        Returns:
            The response from the agent.
        """
        request = self._create_search_request(user_query)
        response = self._client.search(request)
        return response.summary.summary_text
