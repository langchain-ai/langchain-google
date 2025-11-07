# Prerequisites:
# 1. Create a Google Cloud project
# 2. Enable the Google Drive API:
#   https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com
# 3. Authorize credentials for desktop app:
#   https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application # noqa: E501
# 4. For service accounts visit
#   https://cloud.google.com/iam/docs/service-accounts-create

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pydantic import BaseModel, field_validator, model_validator


class GoogleDriveLoader(BaseLoader, BaseModel):
    """Load documents from Google Drive.

    Inherits from [`BaseLoader`][langchain_core.document_loaders.BaseLoader].

    Supports loading from folders, specific documents, or file IDs with authentication.

    !!! note "Installation"

        Requires additional dependencies:

        ```bash
        pip install langchain-google-community[drive]
        ```
    """

    # Generated from https://developers.google.com/drive/api/guides/api-specific-auth
    # limiting to the scopes that are required to read the files
    VALID_SCOPES: ClassVar[Tuple[str, ...]] = (
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.meet.readonly",
        "https://www.googleapis.com/auth/drive.metadata.readonly",
        "https://www.googleapis.com/auth/drive.metadata",
    )

    service_account_key: Path = Path.home() / ".credentials" / "keys.json"
    """Path to the service account key file."""

    credentials_path: Path = Path.home() / ".credentials" / "credentials.json"
    """Path to the credentials file."""

    token_path: Path = Path.home() / ".credentials" / "token.json"
    """Path to the token file."""

    credentials: Any = None
    """Your own google credentials created via your own mechanism"""

    folder_id: Optional[str] = None
    """The folder ID to load from."""

    document_ids: Optional[List[str]] = None
    """The document IDs to load from."""

    file_ids: Optional[List[str]] = None
    """The file IDs to load from."""

    recursive: bool = False
    """Whether to load recursively. Only applies when `folder_id` is given."""

    file_types: Optional[Sequence[str]] = None
    """The file types to load. Only applies when `folder_id` is given."""

    load_trashed_files: bool = False
    """Whether to load trashed files. Only applies when `folder_id` is given."""
    # NOTE(MthwRobinson) - changing the file_loader_cls to type here currently
    # results in pydantic validation errors

    file_loader_cls: Any = None
    """The file loader class to use."""

    file_loader_kwargs: Dict["str", Any] = {}
    """The file loader kwargs to use."""

    load_auth: bool = False
    """Whether to load authorization identities."""

    load_extended_metadata: bool = False
    """Whether to load extended metadata."""

    scopes: List[str] = ["https://www.googleapis.com/auth/drive.file"]
    """The credential scopes to use for Google Drive API access. Default is
    `drive.file` scope."""

    def _get_file_size_from_id(self, id: str) -> str:
        """Fetch the size of the file."""
        try:
            import googleapiclient.errors  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client` "
                "to load authorization identities."
            ) from exc

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        try:
            file = service.files().get(fileId=id, fields="size").execute()
            return file["size"]
        except googleapiclient.errors.HttpError:
            print(
                f"insufficientFilePermissions: The user does not have sufficient \
                permissions to retrieve size for the file with fileId: {id}"
            )
            return "unknown"
        except Exception as exc:
            print(
                f"Error occurred while fetching the size for the file with fileId: {id}"
            )
            print(f"Error: {exc}")
            return "unknown"

    def _get_owner_metadata_from_id(self, id: str) -> str:
        """Fetch the owner of the file."""
        try:
            import googleapiclient.errors  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client` "
                "to load authorization identities."
            ) from exc

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        try:
            file = service.files().get(fileId=id, fields="owners").execute()
            return file["owners"][0].get("emailAddress")
        except googleapiclient.errors.HttpError:
            print(
                f"insufficientFilePermissions: The user does not have sufficient \
                permissions to retrieve owner for the file with fileId: {id}"
            )
            return "unknown"
        except Exception as exc:
            print(
                f"Error occurred while fetching the owner for the file with fileId: \
                {id} with error: {exc}"
            )
            return "unknown"

    def _get_file_path_from_id(self, id: str) -> str:
        """Fetch the full path of the file starting from the root."""
        try:
            import googleapiclient.errors  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client` "
                "to load authorization identities."
            ) from exc

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        path = []
        current_id = id
        while True:
            try:
                file = (
                    service.files()
                    .get(fileId=current_id, fields="name, parents")
                    .execute()
                )
                path.append(file["name"])
                if "parents" in file:
                    current_id = file["parents"][0]
                else:
                    break
            except googleapiclient.errors.HttpError:
                print(
                    f"insufficientFilePermissions: The user does not have sufficient\
                    permissions to retrieve path for the file with fileId: {id}"
                )
                break
        path.reverse()
        return "/".join(path)

    def _get_identity_metadata_from_id(self, id: str) -> List[str]:
        """Fetch the list of people having access to ID file."""
        try:
            import googleapiclient.errors  # type: ignore[import]
            from googleapiclient.discovery import build  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "You must run "
                "`pip install --upgrade "
                "google-api-python-client` "
                "to load authorization identities."
            ) from exc

        authorized_identities: list = []
        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)  # Build the service
        try:
            permissions = service.permissions().list(fileId=id).execute()
        except googleapiclient.errors.HttpError:
            print(
                f"insufficientFilePermissions: The user does not have sufficient \
                permissions to retrieve permission for the file with fileId: {id}"
            )
            return authorized_identities
        except Exception as exc:
            print(
                f"Error occurred while fetching the permissions for the file with \
                fileId: {id}"
            )
            print(f"Error: {exc}")
            return authorized_identities

        for perm in permissions.get("permissions", {}):
            email_id = (
                service.permissions()
                .get(fileId=id, permissionId=perm.get("id", ""), fields="emailAddress")
                .execute()
                .get("emailAddress")
            )
            if email_id:
                authorized_identities.append(email_id)

        return authorized_identities

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Any:
        """Validate that either folder_id or document_ids is set, but not both."""
        if values.get("folder_id") and (
            values.get("document_ids") or values.get("file_ids")
        ):
            raise ValueError(
                "Cannot specify both folder_id and document_ids nor "
                "folder_id and file_ids"
            )
        if (
            not values.get("folder_id")
            and not values.get("document_ids")
            and not values.get("file_ids")
        ):
            raise ValueError("Must specify either folder_id, document_ids, or file_ids")

        file_types = values.get("file_types")
        if file_types:
            if values.get("document_ids") or values.get("file_ids"):
                raise ValueError(
                    "file_types can only be given when folder_id is given,"
                    " (not when document_ids or file_ids are given)."
                )
            type_mapping = {
                "document": "application/vnd.google-apps.document",
                "sheet": "application/vnd.google-apps.spreadsheet",
                "pdf": "application/pdf",
                "presentation": "application/vnd.google-apps.presentation",
            }
            allowed_types = list(type_mapping.keys()) + list(type_mapping.values())
            short_names = ", ".join([f"'{x}'" for x in type_mapping.keys()])
            full_names = ", ".join([f"'{x}'" for x in type_mapping.values()])
            for file_type in file_types:
                if file_type not in allowed_types:
                    raise ValueError(
                        f"Given file type {file_type} is not supported. "
                        f"Supported values are: {short_names}; and "
                        f"their full-form names: {full_names}"
                    )

            # replace short-form file types by full-form file types
            def full_form(x: str) -> str:
                return type_mapping[x] if x in type_mapping else x

            values["file_types"] = [full_form(file_type) for file_type in file_types]
        return values

    @field_validator("credentials_path")
    def validate_credentials_path(cls, v: Any, **kwargs: Any) -> Any:
        """Validate that credentials_path exists."""
        if not v.exists():
            raise ValueError(f"credentials_path {v} does not exist")
        return v

    @field_validator("scopes")
    def validate_scopes(cls, v: List[str]) -> List[str]:
        """Validate that the provided scopes are not empty and
        are valid Google Drive API scopes."""
        if not v:
            raise ValueError("At least one scope must be provided")

        invalid_scopes = [scope for scope in v if scope not in cls.VALID_SCOPES]
        if invalid_scopes:
            raise ValueError(
                f"Invalid Google Drive API scope(s): {', '.join(invalid_scopes)}. "
                f"Valid scopes are: {', '.join(cls.VALID_SCOPES)}"
            )

        return v

    def _load_credentials(self) -> Any:
        """Load credentials."""
        # Adapted from https://developers.google.com/drive/api/v3/quickstart/python
        try:
            from google.auth import default  # type: ignore[import]
            from google.auth.transport.requests import Request  # type: ignore[import]
            from google.oauth2 import service_account  # type: ignore[import]
            from google.oauth2.credentials import Credentials  # type: ignore[import]
            from google_auth_oauthlib.flow import (  # type: ignore[import]
                InstalledAppFlow,
            )
        except ImportError:
            raise ImportError(
                "Could execute GoogleDriveLoader. "
                "Please, install drive dependency group: "
                "`pip install langchain-google-community[drive]`"
            )

        creds = None
        if self.service_account_key.exists():
            return service_account.Credentials.from_service_account_file(
                str(self.service_account_key), scopes=self.scopes
            )

        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self.token_path), self.scopes
            )

        if self.credentials:
            # use whatever was passed to us
            creds = self.credentials
            return creds

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                creds, project = default()
                creds = creds.with_scopes(self.scopes)
                # no need to write to file
                if creds:
                    return creds
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), self.scopes
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def _load_sheet_from_id(self, id: str) -> List[Document]:
        """Load a sheet and all tabs from an ID."""

        from googleapiclient.discovery import build  # type: ignore[import]

        creds = self._load_credentials()
        sheets_service = build("sheets", "v4", credentials=creds)
        spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=id).execute()
        sheets = spreadsheet.get("sheets", [])
        if self.load_auth:
            authorized_identities = self._get_identity_metadata_from_id(id)
        if self.load_extended_metadata:
            owner = self._get_owner_metadata_from_id(id)
            size = self._get_file_size_from_id(id)
            full_path = self._get_file_path_from_id(id)

        documents = []
        for sheet in sheets:
            sheet_name = sheet["properties"]["title"]
            result = (
                sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=id, range=sheet_name)
                .execute()
            )
            values = result.get("values", [])
            if not values:
                continue  # empty sheet

            header = values[0]
            for i, row in enumerate(values[1:], start=1):
                metadata = {
                    "source": (
                        f"https://docs.google.com/spreadsheets/d/{id}/"
                        f"edit?gid={sheet['properties']['sheetId']}"
                    ),
                    "title": f"{spreadsheet['properties']['title']} - {sheet_name}",
                    "row": i,
                }
                if self.load_auth:
                    metadata["authorized_identities"] = authorized_identities
                if self.load_extended_metadata:
                    metadata["owner"] = owner
                    metadata["size"] = size
                    metadata["full_path"] = full_path
                content = []
                for j, v in enumerate(row):
                    title = header[j].strip() if len(header) > j else ""
                    content.append(f"{title}: {v.strip()}")

                page_content = "\n".join(content)
                documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def _load_document_from_id(self, id: str) -> Document:
        """Load a document from an ID."""
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError  # type: ignore[import]
        from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import]

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        if self.load_auth:
            authorized_identities = self._get_identity_metadata_from_id(id)
        if self.load_extended_metadata:
            owner = self._get_owner_metadata_from_id(id)
            size = self._get_file_size_from_id(id)
            full_path = self._get_file_path_from_id(id)

        file = (
            service.files()
            .get(
                fileId=id,
                supportsAllDrives=True,
                fields="modifiedTime,name,webViewLink",
            )
            .execute()
        )
        request = service.files().export_media(fileId=id, mimeType="text/plain")
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        try:
            while done is False:
                status, done = downloader.next_chunk()

        except HttpError as e:
            if e.resp.status == 404:
                print("File not found: {}".format(id))  # noqa: T201
            else:
                print("An error occurred: {}".format(e))  # noqa: T201

        text = fh.getvalue().decode("utf-8")
        metadata = {
            "source": f"{file.get('webViewLink')}",
            "title": f"{file.get('name')}",
            "when": f"{file.get('modifiedTime')}",
        }
        if self.load_auth:
            metadata["authorized_identities"] = authorized_identities  # type: ignore
        if self.load_extended_metadata:
            metadata["owner"] = owner
            metadata["size"] = size
            metadata["full_path"] = full_path
        return Document(page_content=text, metadata=metadata)

    def _load_documents_from_folder(
        self, folder_id: str, *, file_types: Optional[Sequence[str]] = None
    ) -> List[Document]:
        """Load documents from a folder."""
        from googleapiclient.discovery import build

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)
        files = self._fetch_files_recursive(service, folder_id)
        # If file types filter is provided, we'll filter by the file type.
        if file_types:
            _files = [f for f in files if f["mimeType"] in file_types]  # type: ignore
        else:
            _files = files

        returns = []
        for file in _files:
            if file["trashed"] and not self.load_trashed_files:
                continue
            elif file["mimeType"] in [
                "application/vnd.google-apps.document",
                "application/vnd.google-apps.presentation",
            ]:
                returns.append(self._load_document_from_id(file["id"]))  # type: ignore
            elif file["mimeType"] == "application/vnd.google-apps.spreadsheet":
                returns.extend(self._load_sheet_from_id(file["id"]))  # type: ignore
            elif (
                file["mimeType"] == "application/pdf"
                or self.file_loader_cls is not None
            ):
                returns.extend(self._load_file_from_id(file["id"]))  # type: ignore
            else:
                pass
        return returns

    def _fetch_files_recursive(
        self, service: Any, folder_id: str
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """Fetch all files and subfolders recursively."""
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="nextPageToken, files(id, name, mimeType, parents, trashed)",
            )
            .execute()
        )
        files = results.get("files", [])
        returns = []
        for file in files:
            if file["mimeType"] == "application/vnd.google-apps.folder":
                if self.recursive:
                    returns.extend(self._fetch_files_recursive(service, file["id"]))
            else:
                returns.append(file)

        return returns

    def _load_documents_from_ids(self) -> List[Document]:
        """Load documents from a list of IDs."""
        if not self.document_ids:
            raise ValueError("document_ids must be set")

        return [self._load_document_from_id(doc_id) for doc_id in self.document_ids]

    def _load_file_from_id(self, id: str) -> List[Document]:
        """Load a file from an ID."""
        from io import BytesIO

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        creds = self._load_credentials()
        service = build("drive", "v3", credentials=creds)

        if self.load_auth:
            authorized_identities = self._get_identity_metadata_from_id(id)
        if self.load_extended_metadata:
            owner = self._get_owner_metadata_from_id(id)
            size = self._get_file_size_from_id(id)
            full_path = self._get_file_path_from_id(id)

        file = service.files().get(fileId=id, supportsAllDrives=True).execute()
        request = service.files().get_media(fileId=id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        if self.file_loader_cls is not None:
            fh.seek(0)
            loader = self.file_loader_cls(file=fh, **self.file_loader_kwargs)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = f"https://drive.google.com/file/d/{id}/view"
                if "title" not in doc.metadata:
                    doc.metadata["title"] = f"{file.get('name')}"
                if self.load_auth:
                    doc.metadata["authorized_identities"] = authorized_identities
                if self.load_extended_metadata:
                    doc.metadata["owner"] = owner
                    doc.metadata["size"] = size
                    doc.metadata["full_path"] = full_path
            return docs

        else:
            from PyPDF2 import PdfReader  # type: ignore[import]

            content = fh.getvalue()
            pdf_reader = PdfReader(BytesIO(content))

            docs = []
            for i, page in enumerate(pdf_reader.pages):
                metadata = {
                    "source": f"https://drive.google.com/file/d/{id}/view",
                    "title": f"{file.get('name')}",
                    "page": i,
                }
                if self.load_auth:
                    metadata["authorized_identities"] = authorized_identities
                if self.load_extended_metadata:
                    metadata["owner"] = owner
                    metadata["size"] = size
                    metadata["full_path"] = full_path
                docs.append(
                    Document(
                        page_content=page.extract_text(),
                        metadata=metadata,
                    )
                )
            return docs

    def _load_file_from_ids(self) -> List[Document]:
        """Load files from a list of IDs."""
        if not self.file_ids:
            raise ValueError("file_ids must be set")
        docs = []
        for file_id in self.file_ids:
            docs.extend(self._load_file_from_id(file_id))
        return docs

    def load(self) -> List[Document]:
        """Load documents."""
        if self.folder_id:
            return self._load_documents_from_folder(
                self.folder_id, file_types=self.file_types
            )
        elif self.document_ids:
            return self._load_documents_from_ids()
        else:
            return self._load_file_from_ids()
