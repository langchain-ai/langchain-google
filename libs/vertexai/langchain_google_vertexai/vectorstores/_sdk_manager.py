from typing import TYPE_CHECKING, Any

from google.cloud import aiplatform, storage
from google.cloud.aiplatform import telemetry
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.oauth2.service_account import Credentials

if TYPE_CHECKING:
    from google.cloud import datastore  # type: ignore[attr-defined, unused-ignore]

from langchain_google_vertexai._utils import get_client_info, get_user_agent


class VectorSearchSDKManager:
    """Class in charge of building all Google Cloud SDK Objects needed to build
    VectorStores from `project_id`, credentials or other specifications.

    Abstracts away the authentication layer.
    """

    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        api_version: str = "v1",
        credentials: Credentials | None = None,
        credentials_path: str | None = None,
    ) -> None:
        """Constructor.

        If `credentials` is provided, those credentials are used. If not provided
        `credentials_path` is used to retrieve credentials from a file. If also not
        provided, falls back to default credentials.

        Args:
            project_id: Id of the project.
            region: Region of the project. e.g. `'us-central1'`
            credentials: Google cloud Credentials object.
            credentials_path: Google Cloud Credentials json file path.
        """
        self._project_id = project_id
        self._region = region
        self._api_version = api_version

        if credentials is not None:
            self._credentials: Credentials | None = credentials
        elif credentials_path is not None:
            self._credentials = Credentials.from_service_account_file(credentials_path)
        else:
            self._credentials = None

        self.initialize_aiplatform()

    def initialize_aiplatform(self) -> None:
        """Initializes `aiplatform`."""
        aiplatform.init(
            project=self._project_id,
            location=self._region,
            credentials=self._credentials,
        )

    def get_gcs_client(self) -> storage.Client:
        """Retrieves a Google Cloud Storage client.

        Returns:
            Google Cloud Storage Agent.
        """
        return storage.Client(
            project=self._project_id,
            credentials=self._credentials,
            client_info=get_client_info(module="vertex-ai-matching-engine"),
        )

    def get_gcs_bucket(self, bucket_name: str) -> storage.Bucket:
        """Retrieves a Google Cloud Bucket by bucket name.

        Args:
            bucket_name: Name of the bucket to be retrieved.

        Returns:
            Google Cloud Bucket.
        """
        client = self.get_gcs_client()
        return client.get_bucket(bucket_name)

    def get_index(self, index_id: str) -> MatchingEngineIndex:
        """Retrieves a `MatchingEngineIndex` (`VectorSearchIndex`) by ID.

        Args:
            index_id: ID of the index to be retrieved.

        Returns:
            `MatchingEngineIndex` instance.
        """
        _, user_agent = get_user_agent("vertex-ai-matching-engine")
        with telemetry.tool_context_manager(user_agent):
            return MatchingEngineIndex(
                index_name=index_id,
                project=self._project_id,
                location=self._region,
                credentials=self._credentials,
            )

    def get_collection(self, collection_id: str) -> Any:
        """Retrieves a Vector Search V2 Collection by ID.

        Args:
            collection_id: The ID of the collection.

        Returns:
            A SimpleNamespace object containing the collection's resource name.
        """
        from types import SimpleNamespace

        collection = SimpleNamespace()
        collection.resource_name = (
            f"projects/{self._project_id}/locations/{self._region}/"
            f"collections/{collection_id}"
        )
        collection.location = self._region
        return collection

    def get_endpoint(self, endpoint_id: str) -> MatchingEngineIndexEndpoint:
        """Retrieves a `MatchingEngineIndexEndpoint` (`VectorSearchIndexEndpoint`) by ID.

        Args:
            endpoint_id: ID of the endpoint to be retrieved.

        Returns:
            `MatchingEngineIndexEndpoint` instance.
        """  # noqa: E501
        _, user_agent = get_user_agent("vertex-ai-matching-engine")
        with telemetry.tool_context_manager(user_agent):
            return MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_id,
                project=self._project_id,
                location=self._region,
                credentials=self._credentials,
            )

    def get_datastore_client(self, **kwargs: Any) -> "datastore.Client":
        """Gets a `datastore` Client.

        Args:
            **kwargs: Keyword arguments to pass to `datastore.Client` constructor.

        Returns:
            `datastore` Client.
        """
        from google.cloud import datastore  # type: ignore[attr-defined, unused-ignore]

        return datastore.Client(
            project=self._project_id,
            credentials=self._credentials,
            client_info=get_client_info(module="vertex-ai-matching-engine"),
            **kwargs,
        )

    def get_v2_client(self) -> dict[str, Any]:
        """Get V2 clients for Vector Search 2.0 operations.

        Returns:
            Dictionary containing V2 clients:
                - data_object_service_client: For CRUD operations on data objects
                - data_object_search_service_client: For search/query operations

        Raises:
            ImportError: If google-cloud-vectorsearch is not installed.
        """
        try:
            from google.cloud import vectorsearch_v1beta
        except ImportError as e:
            msg = (
                "google-cloud-vectorsearch is not installed. "
                "Install it with: pip install google-cloud-vectorsearch"
            )
            raise ImportError(msg) from e

        return {
            "data_object_service_client": (
                vectorsearch_v1beta.DataObjectServiceClient(
                    credentials=self._credentials
                )
            ),
            "data_object_search_service_client": (
                vectorsearch_v1beta.DataObjectSearchServiceClient(
                    credentials=self._credentials
                )
            ),
        }
