from typing import Optional

from google.auth.credentials import Credentials  # type: ignore
from google.cloud import discoveryengine_v1alpha


class VertexRankSDKManager:
    def __init__(
        self,
        project_id: str,
        location_id: str,
        credentials: Optional[Credentials] = None,
        credentials_path: Optional[str] = None,
    ):
        """
        Initializes the VertexRankSDKManager with the given project ID,
        location ID, and credentials.

        Args:
            project_id: The Google Cloud project ID.
            location_id: The location ID for the ranking service.
            credentials: The Google Cloud credentials object (optional).
            credentials_path: The path to the Google Cloud service account
            credentials file (optional).
        """
        self.project_id = project_id
        self.location_id = location_id
        self.credentials = (
            credentials or Credentials.from_service_account_file(credentials_path)
            if credentials_path
            else None
        )

    def get_rank_service_client(self) -> discoveryengine_v1alpha.RankServiceClient:
        """
        Returns a RankServiceClient instance for making API calls to the
        Vertex AI Ranking service.

        Returns:
            A RankServiceClient instance.
        """
        return discoveryengine_v1alpha.RankServiceClient(credentials=self.credentials)
