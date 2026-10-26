"""Local document indexing. Callers must serialize writes for the same source."""

import json
import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "localbrain"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 8
SUPPORTED_SUFFIXES = {".txt", ".md"}


def _pending_sources(collection: Collection) -> set[str]:
    pending = json.loads((collection.metadata or {}).get("pending_sources", "[]"))
    if not isinstance(pending, list) or any(not isinstance(source, str) for source in pending):
        raise RuntimeError("Некорректная метка незавершённой индексации")
    return set(pending)


def _set_source_pending(collection: Collection, source_id: str, pending: bool) -> None:
    sources = _pending_sources(collection)
    if (source_id in sources) == pending:
        return
    if pending:
        sources.add(source_id)
    else:
        sources.discard(source_id)
    metadata = dict(collection.metadata or {})
    if sources:
        metadata["pending_sources"] = json.dumps(sorted(sources))
    else:
        metadata.pop("pending_sources", None)
    collection.modify(metadata=metadata)


def ensure_index_consistent(collection: Collection) -> None:
    if _pending_sources(collection):
        raise RuntimeError("Восстановление индекса не завершено; повторите индексацию источника")


@dataclass
class IngestResult:
    files: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@contextmanager
def open_collection(settings: Settings) -> Iterator[Collection]:
    """Disable Chroma's default embedding function and anonymous telemetry."""
    with chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    ) as client:
        collection = client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            metadata={"embedding_model": settings.ollama_embedding_model},
        )
        if (collection.metadata or {}).get("embedding_model") != settings.ollama_embedding_model:
            raise ValueError(
                "Коллекция ChromaDB использует другую embedding-модель. "
                "Для новой модели укажите отдельный CHROMA_PERSIST_DIR."
            )
        yield collection


def split_text(text: str) -> list[str]:
    if not text.strip():
        return []
    # Character-only separators give exactly 100 characters of overlap,
    # including whitespace, instead of a paragraph-dependent overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[""],
        strip_whitespace=False,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def file_source_id(path: Path) -> str:
    """Same file and directory produce the same identity across process restarts."""
    identity = os.path.normcase(str(path.resolve()))
    return "file:" + sha256(identity.encode()).hexdigest()


def index_document(
    text: str,
    *,
    source: str,
    source_id: str,
    collection: Collection,
    embeddings: Embeddings,
) -> int:
    """Index UTF-8 text for one caller-owned source; never prune other sources.

    Embeddings are calculated before writing, so an embedding failure preserves
    the old document. On a write exception restore its previous records before
    releasing the caller's single-writer lock. This is not crash atomicity.
    """
    if not source or not source_id:
        raise ValueError("Имя и идентификатор источника не должны быть пустыми")
    chunks = split_text(text)
    ids = [f"{source_id}:{index}" for index in range(len(chunks))]
    content_hash = sha256(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}:{text}".encode()).hexdigest()
    was_pending = source_id in _pending_sources(collection)
    previous = collection.get(
        where={"source_id": source_id}, include=["metadatas", "documents", "embeddings"]
    )
    if set(previous["ids"]) == set(ids) and all(
        metadata
        and metadata.get("content_hash") == content_hash
        and metadata.get("source") == source
        for metadata in previous["metadatas"] or []
    ):
        logger.info("Без изменений: %s (%s чанков)", source, len(chunks))
        _set_source_pending(collection, source_id, False)
        return len(chunks)

    vectors: list[list[float]] = []
    dimension = None
    # Keep only one document in memory and make small sequential Ollama requests.
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_vectors = embeddings.embed_documents(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError("Ollama вернула неверное количество embedding-векторов")
        for vector in batch_vectors:
            if dimension is None:
                dimension = len(vector)
            if (
                not vector
                or len(vector) != dimension
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError("Ollama вернула некорректный embedding-вектор")
        vectors.extend(batch_vectors)

    # Persist the marker before the first record changes. A failed recovery or
    # process exit leaves searches blocked, even after reopening the database.
    _set_source_pending(collection, source_id, True)
    try:
        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=vectors[start:end],
                metadatas=[
                    {
                        "source": source,
                        "source_id": source_id,
                        "chunk_index": index,
                        "content_hash": content_hash,
                    }
                    for index in range(start, min(end, len(chunks)))
                ],
            )
        current_ids = set(ids)
        stale_ids = [old_id for old_id in previous["ids"] if old_id not in current_ids]
        for start in range(0, len(stale_ids), BATCH_SIZE):
            collection.delete(ids=stale_ids[start : start + BATCH_SIZE])
    except Exception:
        # Remove the incomplete replacement, then restore the complete snapshot.
        collection.delete(where={"source_id": source_id})
        for start in range(0, len(previous["ids"]), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.upsert(
                ids=previous["ids"][start:end],
                documents=previous["documents"][start:end],
                metadatas=previous["metadatas"][start:end],
                embeddings=previous["embeddings"][start:end],
            )
        # A snapshot from an earlier failed recovery may already be incomplete.
        # Restoring it must not make that source searchable again.
        _set_source_pending(collection, source_id, was_pending)
        raise
    _set_source_pending(collection, source_id, False)
    logger.info("Проиндексирован: %s (%s чанков)", source, len(chunks))
    return len(chunks)


def ingest_directory(
    directory: Path, *, collection: Collection, embeddings: Embeddings
) -> IngestResult:
    """Read only immediate regular .txt/.md files; leave absent sources untouched."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Каталог документов не найден: проверьте DOCUMENTS_DIR")
    result = IngestResult()
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            reason = (
                "некорректный UTF-8" if isinstance(exc, UnicodeDecodeError) else "ошибка чтения"
            )
            result.errors[path.name] = reason
            logger.warning("Пропущен %s: %s; прежние чанки сохранены", path.name, reason)
            continue
        result.files[path.name] = index_document(
            text,
            source=path.name,
            source_id=file_source_id(path),
            collection=collection,
            embeddings=embeddings,
        )
    return result
