from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from langchain_core.documents import Document

if TYPE_CHECKING:
    from google.cloud import datastore  # type: ignore[attr-defined, unused-ignore]


class DocumentStorage(ABC):
    """Abstract interface of a key, text storage for retrieving documents."""

    @abstractmethod
    def get_by_id(self, document_id: str) -> Document | None:
        """Gets  a document by its id. If not found, returns None.
        Args:
            document_id: Id of the document to get from the storage.
        Returns:
            Document if found, otherwise None.
        """
        raise NotImplementedError()

    @abstractmethod
    def store_by_id(self, document_id: str, document: Document):
        """Stores a document associated to a document_id.
        Args:
            document_id: Id of the document to be stored.
            document: Document to be stored.
        """
        raise NotImplementedError()

    def batch_store_by_id(self, ids: List[str], documents: List[Document]) -> None:
        """Stores a list of ids and documents in batch.
        The default implementation only loops to the individual `store_by_id`.
        Subclasses that have faster ways to store data via batch uploading should
        implement the proper way.
        Args:
            ids: List of ids for the text.
            documents: List of documents.
        """
        for id_, document in zip(ids, documents):
            self.store_by_id(id_, document)

    def batch_get_by_id(self, ids: List[str]) -> List[Document | None]:
        """Gets a batch of documents by id.
        The default implementation only loops `get_by_id`.
        Subclasses that have faster ways to retrieve data by batch should implement
        this method.
        Args:
            ids: List of ids for the text.
        Returns:
            List of documents. If the key id is not found for any id record returns a
                None instead.
        """
        return [self.get_by_id(id_) for id_ in ids]


class GCSDocumentStorage(DocumentStorage):
    """Stores documents in Google Cloud Storage.
    For each pair id, document_text the name of the blob will be {prefix}/{id} stored
    in plain text format.
    """

    def __init__(
        self, bucket: storage.Bucket, prefix: Optional[str] = "documents"
    ) -> None:
        """Constructor.
        Args:
            bucket: Bucket where the documents will be stored.
            prefix: Prefix that is prepended to all document names.
        """
        super().__init__()
        self._bucket = bucket
        self._prefix = prefix

    def get_by_id(self, document_id: str) -> Document | None:
        """Gets the text of a document by its id. If not found, returns None.
        Args:
            document_id: Id of the document to get from the storage.
        Returns:
            Document if found, otherwise None.
        """

        blob_name = self._get_blob_name(document_id)
        existing_blob = self._bucket.get_blob(blob_name)

        if existing_blob is None:
            return None

        document_str = existing_blob.download_as_text()
        document_json: Dict[str, Any] = json.loads(document_str)
        return Document(**document_json)

    def store_by_id(self, document_id: str, document: Document) -> None:
        """Stores a document text associated to a document_id.
        Args:
            document_id: Id of the document to be stored.
            document: Document to be stored.
        """
        blob_name = self._get_blob_name(document_id)
        new_blow = self._bucket.blob(blob_name)

        document_json = document.dict()
        document_text = json.dumps(document_json)
        new_blow.upload_from_string(document_text)

    def _get_blob_name(self, document_id: str) -> str:
        """Builds a blob name using the prefix and the document_id.
        Args:
            document_id: Id of the document.
        Returns:
            Name of the blob that the document will be/is stored in
        """
        return f"{self._prefix}/{document_id}"


class DataStoreDocumentStorage(DocumentStorage):
    """Stores documents in Google Cloud DataStore."""

    def __init__(
        self,
        datastore_client: datastore.Client,
        kind: str = "document_id",
        text_property_name: str = "text",
        metadata_property_name: str = "metadata",
    ) -> None:
        """Constructor.
        Args:
            bucket: Bucket where the documents will be stored.
            prefix: Prefix that is prepended to all document names.
        """
        super().__init__()
        self._client = datastore_client
        self._text_property_name = text_property_name
        self._metadata_property_name = metadata_property_name
        self._kind = kind

    def get_by_id(self, document_id: str) -> Document | None:
        """Gets the text of a document by its id. If not found, returns None.
        Args:
            document_id: Id of the document to get from the storage.
        Returns:
            Text of the document if found, otherwise None.
        """
        key = self._client.key(self._kind, document_id)
        entity = self._client.get(key)

        if entity is None:
            return None

        return Document(
            page_content=entity[self._text_property_name],
            metadata=self._convert_entity_to_dict(entity[self._metadata_property_name]),
        )

    def store_by_id(self, document_id: str, document: Document) -> None:
        """Stores a document text associated to a document_id.
        Args:
            document_id: Id of the document to be stored.
            text: Text of the document to be stored.
        """
        with self._client.transaction():
            key = self._client.key(self._kind, document_id)

            entity = self._client.entity(key=key)
            entity[self._text_property_name] = document.page_content
            entity[self._metadata_property_name] = document.metadata

            self._client.put(entity)

    def batch_get_by_id(self, ids: List[str]) -> List[Document | None]:
        """Gets a batch of documents by id.
        Args:
            ids: List of ids for the text.
        Returns:
            List of texts. If the key id is not found for any id record returns a None
                instead.
        """
        keys = [self._client.key(self._kind, id_) for id_ in ids]

        # TODO: Handle when a key is not present
        entities = self._client.get_multi(keys)

        # Entities are not sorted by key by default, the order is unclear. This orders
        # the list by the id retrieved.
        entity_id_lookup = {entity.key.id_or_name: entity for entity in entities}
        entities = [entity_id_lookup[id_] for id_ in ids]

        return [
            Document(
                page_content=entity[self._text_property_name],
                metadata=self._convert_entity_to_dict(
                    entity[self._metadata_property_name]
                ),
            )
            for entity in entities
        ]

    def batch_store_by_id(self, ids: List[str], documents: List[Document]) -> None:
        """Stores a list of ids and documents in batch.
        Args:
            ids: List of ids for the text.
            texts: List of texts.
        """

        with self._client.transaction():
            keys = [self._client.key(self._kind, id_) for id_ in ids]

            entities = []
            for key, document in zip(keys, documents):
                entity = self._client.entity(key=key)
                entity[self._text_property_name] = document.page_content
                entity[self._metadata_property_name] = document.metadata
                entities.append(entity)

            self._client.put_multi(entities)

    def _convert_entity_to_dict(self, entity: datastore.Entity) -> Dict[str, Any]:
        """Recursively transform an entity into a plain dictionary."""
        from google.cloud import datastore  # type: ignore[attr-defined, unused-ignore]

        dict_entity = dict(entity)
        for key in dict_entity:
            value = dict_entity[key]
            if isinstance(value, datastore.Entity):
                dict_entity[key] = self._convert_entity_to_dict(value)
        return dict_entity
