from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool


class VertexSearchTool(BaseTool):
    """Class that exposes a tool to interface with an App in Vertex Search and
    Conversation. Now Agent Builder.
    """

    project_id: str
    """ Id of the GCP project the App is in"""

    engine_id: str
    """ Id of the Vertex Search App"""

    location: str = "global"
    """ Location of the Vertex Search App. Defaults to `global`"""

    return_snippet: bool = True
    """ Whether to return a snippet of the search results"""

    max_number_of_documents: int = 10
    """ Maximum number of documents to retrieve in the search"""

    summary_result_count: int = 3
    """ Number of documents to include in the summary"""

    summary_include_citations: bool = True
    """ Whether to include citations in the summary """

    summary_prompt: Optional[str] = None

    summary_spec_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Additional keyword arguments for 
        `discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec`
    """
    search_request_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Additional keyword arguments for 
        `discoveryengine.SearchRequest`
    """

    content_search_spec_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Additional keyword arguments for 
        `discoveryengine.ContentSearchSpec`
    """

    def _run(self, search_query: str, **kwargs: Any) -> str:
        """Runs the tool.

        Args:
            search_query: The query to run by the agent.

        Returns:
            The response from the agent.
        """

        import google.cloud.discoveryengine_v1 as discoveryengine  # type: ignore[import-untyped, unused-ignore]
        from google.api_core.client_options import ClientOptions

        client_options_kwargs = {}

        if self.location != "global":
            client_options_kwargs = dict(
                api_endpoint=f"{self.location}-discoveryengine.googleapis.com"
            )

        client_options = ClientOptions(**client_options_kwargs)

        client = discoveryengine.SearchServiceClient(client_options=client_options)
        serving_config = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/collections/default_collection/engines/{self.engine_id}"
            "/servingConfigs/default_config"
        )

        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=self.return_snippet
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=self.summary_result_count,
                include_citations=self.summary_include_citations,
                model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                    preamble=self.summary_prompt
                ),
                **self.summary_spec_kwargs,
            ),
            **self.content_search_spec_kwargs,
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=search_query,
            page_size=self.max_number_of_documents,
            content_search_spec=content_search_spec,
            **self.search_request_kwargs,
        )

        response = client.search(request)

        return response.summary.summary_text
