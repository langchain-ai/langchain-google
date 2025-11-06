"""Util that calls Google Search."""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator


class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Custom Search API.

    Performs web searches using Google Custom Search API and returns results
    with snippets, titles, and links.

    !!! note "Setup Required"

        1. Enable [Custom Search API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)
        2. Create API key in [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
        3. Create custom search engine at [Programmable Search Engine](https://programmablesearchengine.google.com)
        4. Set `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` environment variables
    """

    search_engine: Any = None

    google_api_key: Optional[str] = None
    """Google API key for authentication."""

    google_cse_id: Optional[str] = None
    """Custom Search Engine ID."""

    k: int = 10
    """Number of results to return."""

    siterestrict: bool = False
    """Whether to restrict search to specific sites."""

    model_config = ConfigDict(
        extra="forbid",
    )

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        cse = self.search_engine.cse()
        if self.siterestrict:
            cse = cse.siterestrict()
        res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
        return res.get("items", [])

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        try:
            from googleapiclient.discovery import build  # type: ignore[import]

        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with `pip install langchain-google-community`"
            )

        service = build("customsearch", "v1", developerKey=google_api_key)
        values["search_engine"] = service

        return values

    def run(self, query: str) -> str:
        """Run query through Google Search and parse result."""
        snippets = []
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(
        self,
        query: str,
        num_results: int,
        search_params: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Run query through Google Search and return metadata.

        Args:
            query: The query to search for.
            num_results: Number of results to return.
            search_params: Additional search parameters.

        Returns:
            Search results with snippet, title, and link for each result.
        """
        metadata_results = []
        results = self._google_search_results(
            query, num=num_results, **(search_params or {})
        )
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_results.append(metadata_result)

        return metadata_results


class GoogleSearchRun(BaseTool):
    """Tool that queries the Google Custom Search API.

    Inherits from [`BaseTool`][langchain_core.tools.BaseTool].

    Returns concatenated snippets from search results.
    """

    name: str = "google_search"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: GoogleSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the Google search.

        Args:
            query: Search query string.
            run_manager: Optional callback manager.

        Returns:
            Concatenated snippets from search results.
        """
        return self.api_wrapper.run(query)


class GoogleSearchResults(BaseTool):
    """Tool that queries the Google Custom Search API and returns JSON.

    Inherits from [`BaseTool`][langchain_core.tools.BaseTool].

    Returns structured search results with metadata.
    """

    name: str = "google_search_results_json"
    description: str = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4
    api_wrapper: GoogleSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the Google search and return structured results.

        Args:
            query: Search query string.
            run_manager: Optional callback manager.

        Returns:
            JSON string of search results with title, link, and snippet.
        """
        return str(self.api_wrapper.results(query, self.num_results))
