from collections.abc import Iterator
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import pytest
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings

import ingest
from app.config import Settings
from app.ingestion import (
    BATCH_SIZE,
    ensure_index_consistent,
    file_source_id,
    index_document,
    ingest_directory,
    open_collection,
    split_text,
)


def fake_vectors(texts: list[str]) -> list[list[float]]:
    return [
        [float(len(text)), float(int(sha256(text.encode()).hexdigest()[:4], 16)), 1.0]
        for text in texts
    ]


@pytest.fixture
def embeddings() -> Mock:
    mock = Mock(spec=Embeddings)
    mock.embed_documents.side_effect = fake_vectors
    return mock


@pytest.fixture
def collection(tmp_path: Path) -> Iterator[Collection]:
    with open_collection(Settings(chroma_persist_dir=tmp_path / "chroma")) as collection:
        yield collection


def test_character_chunk_size_overlap_and_whitespace() -> None:
    text = "\n  " + "".join(chr(0x400 + index % 300) for index in range(1197))
    chunks = split_text(text)
    assert [len(chunk) for chunk in chunks] == [500, 500, 400]
    assert chunks == [text[:500], text[400:900], text[800:]]
    assert chunks[0][-100:] == chunks[1][:100]
    assert chunks[1][-100:] == chunks[2][:100]


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_text_has_no_chunks(text: str) -> None:
    assert split_text(text) == []


def test_long_whitespace_is_filtered_without_stripping_meaningful_chunks(
    collection: Collection, embeddings: Mock
) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = "Начало" + " " * 2000 + "Конец"
    original = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=[""], strip_whitespace=False
    ).split_text(text)
    assert any(not chunk.strip() for chunk in original)
    expected = [chunk for chunk in original if chunk.strip()]
    assert split_text(text) == expected
    assert len(expected) == 2
    # Reproduce the old index, including blank chunks and its old numbering/hash.
    collection.add(
        ids=[f"spaces:{index}" for index in range(len(original))],
        documents=original,
        embeddings=fake_vectors(original),
        metadatas=[
            {
                "source": "spaces.txt",
                "source_id": "spaces",
                "chunk_index": index,
                "content_hash": sha256(f"500:100:{text}".encode()).hexdigest(),
            }
            for index in range(len(original))
        ],
    )
    index_document(
        text, source="spaces.txt", source_id="spaces", collection=collection, embeddings=embeddings
    )
    saved = collection.get(include=["documents", "metadatas"])
    by_id = {item_id: text for item_id, text in zip(saved["ids"], saved["documents"], strict=True)}
    assert by_id == {f"spaces:{index}": chunk for index, chunk in enumerate(expected)}
    assert {metadata["chunk_index"] for metadata in saved["metadatas"]} == {0, 1}


def test_second_write_batch_failure_restores_entire_previous_version(
    collection: Collection, embeddings: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_document(
        "Прежняя версия. " * 350,
        source="demo.txt",
        source_id="demo",
        collection=collection,
        embeddings=embeddings,
    )
    index_document(
        "Другой источник",
        source="other.txt",
        source_id="other",
        collection=collection,
        embeddings=embeddings,
    )
    before = collection.get(include=["documents", "metadatas", "embeddings"])
    upsert = collection.upsert
    calls = 0

    def fail_second(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch failure")
        upsert(**kwargs)

    monkeypatch.setattr(collection, "upsert", fail_second)
    with pytest.raises(RuntimeError, match="second batch"):
        index_document(
            "Новая версия. " * 500,
            source="demo.txt",
            source_id="demo",
            collection=collection,
            embeddings=embeddings,
        )
    after = collection.get(include=["documents", "metadatas", "embeddings"])

    def records(data: dict) -> dict:
        return {
            item_id: (doc, meta, vector.tolist())
            for item_id, doc, meta, vector in zip(
                data["ids"], data["documents"], data["metadatas"], data["embeddings"], strict=True
            )
        }

    assert records(after) == records(before)
    ensure_index_consistent(collection)
    monkeypatch.setattr(collection, "upsert", upsert)
    text = "Новая версия. " * 500
    count = index_document(
        text, source="demo.txt", source_id="demo", collection=collection, embeddings=embeddings
    )
    assert (
        index_document(
            text, source="demo.txt", source_id="demo", collection=collection, embeddings=embeddings
        )
        == count
    )
    assert collection.count() == count + 1


def test_recovery_failure_blocks_search_until_reindex(
    tmp_path: Path, collection: Collection, embeddings: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_document(
        "Старый текст",
        source="demo.txt",
        source_id="demo",
        collection=collection,
        embeddings=embeddings,
    )
    upsert = collection.upsert
    calls = 0

    def fail_update_and_restore(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("persistent write error")
        upsert(**kwargs)

    monkeypatch.setattr(collection, "upsert", fail_update_and_restore)
    with pytest.raises(RuntimeError):
        index_document(
            "Новый текст " * 500,
            source="demo.txt",
            source_id="demo",
            collection=collection,
            embeddings=embeddings,
        )
    with pytest.raises(RuntimeError, match="Восстановление"):
        ensure_index_consistent(collection)
    with open_collection(Settings(chroma_persist_dir=tmp_path / "chroma")) as reopened:
        with pytest.raises(RuntimeError, match="Восстановление"):
            ensure_index_consistent(reopened)

    # A failed retry must retain the durable marker even if its rollback succeeds:
    # the snapshot it restores came from an already incomplete recovery.
    calls = 0

    def fail_retry_second_batch(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("retry write error")
        upsert(**kwargs)

    monkeypatch.setattr(collection, "upsert", fail_retry_second_batch)
    with pytest.raises(RuntimeError, match="retry write"):
        index_document(
            "Новый текст " * 500,
            source="demo.txt",
            source_id="demo",
            collection=collection,
            embeddings=embeddings,
        )
    with open_collection(Settings(chroma_persist_dir=tmp_path / "chroma")) as reopened:
        with pytest.raises(RuntimeError, match="Восстановление"):
            ensure_index_consistent(reopened)

    monkeypatch.setattr(collection, "upsert", upsert)
    index_document(
        "Восстановленный текст",
        source="demo.txt",
        source_id="demo",
        collection=collection,
        embeddings=embeddings,
    )
    ensure_index_consistent(collection)
    with open_collection(Settings(chroma_persist_dir=tmp_path / "chroma")) as reopened:
        ensure_index_consistent(reopened)


def test_reads_only_supported_immediate_files(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    for name in ("first.txt", "second.md", "upper.TXT"):
        (directory / name).write_text("Текст документа", encoding="utf-8")
    (directory / "ignored.pdf").write_bytes(b"not a document")
    for name in ("private", "nested", "directory.txt"):
        nested = directory / name
        nested.mkdir()
        (nested / "secret.md").write_text("Не индексировать", encoding="utf-8")

    result = ingest_directory(directory, collection=collection, embeddings=embeddings)

    assert result.files == {"first.txt": 1, "second.md": 1, "upper.TXT": 1}
    assert not result.errors
    assert collection.count() == 3
    assert {item["source"] for item in collection.get()["metadatas"]} == set(result.files)


def test_external_symbolic_link_is_not_read(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("Приватный текст вне каталога", encoding="utf-8")
    try:
        (directory / "linked.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Creating symbolic links is unavailable: {exc}")

    result = ingest_directory(directory, collection=collection, embeddings=embeddings)

    assert not result.files
    assert collection.count() == 0
    embeddings.embed_documents.assert_not_called()


def test_file_marked_as_symbolic_link_is_skipped(
    tmp_path: Path, collection: Collection, embeddings: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the guard even on Windows accounts without symlink privileges.
    path = tmp_path / "linked.txt"
    path.write_text("Не читать", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    result = ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
    assert not result.files
    assert collection.count() == 0
    embeddings.embed_documents.assert_not_called()


def test_text_metadata_and_numeric_vectors_are_stored(
    collection: Collection, embeddings: Mock
) -> None:
    text = "Осмысленный русский текст. " * 30
    chunks = split_text(text)
    count = index_document(
        text, source="demo.md", source_id="test:demo", collection=collection, embeddings=embeddings
    )
    ids = [f"test:demo:{index}" for index in range(count)]
    saved = collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    positions = {item_id: index for index, item_id in enumerate(saved["ids"])}
    for index, item_id in enumerate(ids):
        position = positions[item_id]
        assert saved["documents"][position] == chunks[index]
        assert saved["metadatas"][position]["source"] == "demo.md"
        assert saved["metadatas"][position]["source_id"] == "test:demo"
        assert saved["metadatas"][position]["chunk_index"] == index
        assert list(saved["embeddings"][position]) == fake_vectors([chunks[index]])[0]
    assert count == len(chunks)


def test_repeat_skips_embeddings_and_keeps_stable_ids(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "demo.txt").write_text("Документ. " * 100, encoding="utf-8")
    first = ingest_directory(directory, collection=collection, embeddings=embeddings)
    before = collection.get(include=["documents", "metadatas"])
    embeddings.reset_mock()

    second = ingest_directory(directory, collection=collection, embeddings=embeddings)

    assert first == second
    assert collection.get(include=["documents", "metadatas"]) == before
    assert collection.count() == sum(first.files.values())
    embeddings.embed_documents.assert_not_called()


def test_update_grow_shrink_and_empty_preserves_other_sources(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    path = tmp_path / "demo.txt"
    path.write_text("a" * 1300, encoding="utf-8")
    source_id = file_source_id(path)
    index_document(
        "Другой источник с тем же именем",
        source=path.name,
        source_id="upload:other",
        collection=collection,
        embeddings=embeddings,
    )
    other = collection.get(where={"source_id": "upload:other"}, include=["documents", "metadatas"])
    ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
    assert collection.count() == 4

    for text in ("b" * 200, "c" * 1700, "d" * 1700, " \n\t"):
        path.write_text(text, encoding="utf-8")
        result = ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
        expected_chunks = split_text(text)
        saved = collection.get(where={"source_id": source_id}, include=["documents"])
        assert result.files[path.name] == len(expected_chunks)
        assert set(saved["ids"]) == {
            f"{source_id}:{index}" for index in range(len(expected_chunks))
        }
        assert sorted(saved["documents"]) == sorted(expected_chunks)
        assert collection.count() == len(expected_chunks) + 1
        assert (
            collection.get(where={"source_id": "upload:other"}, include=["documents", "metadatas"])
            == other
        )


def test_corrupt_utf8_preserves_old_chunks_and_continues(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    path = tmp_path / "broken.txt"
    path.write_text("Прежний корректный текст", encoding="utf-8")
    ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
    before = collection.get()
    path.write_bytes(b"\xff\xfe\x80")
    (tmp_path / "valid.md").write_text("Другой документ", encoding="utf-8")

    result = ingest_directory(tmp_path, collection=collection, embeddings=embeddings)

    assert result.errors == {"broken.txt": "некорректный UTF-8"}
    assert result.files == {"valid.md": 1}
    assert collection.get(ids=before["ids"])["documents"] == before["documents"]
    assert collection.count() == 2


def test_read_error_is_reported(
    tmp_path: Path, collection: Collection, embeddings: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "denied.txt"
    path.write_text("Текст", encoding="utf-8")

    def deny_read(self: Path, **kwargs: object) -> str:
        raise PermissionError("simulated read failure")

    monkeypatch.setattr(Path, "read_text", deny_read)
    result = ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
    assert result.errors == {"denied.txt": "ошибка чтения"}
    embeddings.embed_documents.assert_not_called()


def test_empty_file_and_missing_directory(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    (tmp_path / "empty.md").write_text(" \n ", encoding="utf-8")
    result = ingest_directory(tmp_path, collection=collection, embeddings=embeddings)
    assert result.files == {"empty.md": 0}
    assert collection.count() == 0
    embeddings.embed_documents.assert_not_called()
    with pytest.raises(FileNotFoundError, match="DOCUMENTS_DIR"):
        ingest_directory(tmp_path / "missing", collection=collection, embeddings=embeddings)


def test_embedding_requests_are_batched(collection: Collection, embeddings: Mock) -> None:
    text = "Я" * 7000
    expected = split_text(text)
    index_document(
        text, source="large.txt", source_id="large", collection=collection, embeddings=embeddings
    )
    batches = [call.args[0] for call in embeddings.embed_documents.call_args_list]
    assert len(batches) > 1
    assert all(1 <= len(batch) <= BATCH_SIZE for batch in batches)
    assert [chunk for batch in batches for chunk in batch] == expected


def test_embedding_failure_preserves_previous_document(
    collection: Collection, embeddings: Mock
) -> None:
    index_document(
        "Старый текст",
        source="demo.txt",
        source_id="demo",
        collection=collection,
        embeddings=embeddings,
    )
    before = collection.get()
    embeddings.embed_documents.side_effect = [
        fake_vectors(["x" * 500] * BATCH_SIZE),
        ConnectionError(),
    ]
    with pytest.raises(ConnectionError):
        index_document(
            "x" * 7000,
            source="demo.txt",
            source_id="demo",
            collection=collection,
            embeddings=embeddings,
        )
    assert collection.get() == before


def test_retry_completes_an_interrupted_chroma_write(
    collection: Collection, embeddings: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    upsert = collection.upsert
    calls = 0

    def fail_after_first_batch(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated database failure")
        upsert(**kwargs)

    text = "Повторяемая запись " * 400
    monkeypatch.setattr(collection, "upsert", fail_after_first_batch)
    with pytest.raises(RuntimeError, match="database failure"):
        index_document(
            text,
            source="retry.txt",
            source_id="retry",
            collection=collection,
            embeddings=embeddings,
        )
    assert collection.count() == 0
    monkeypatch.setattr(collection, "upsert", upsert)
    count = index_document(
        text, source="retry.txt", source_id="retry", collection=collection, embeddings=embeddings
    )
    assert collection.count() == count == len(split_text(text))


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[], []],
        [[1.0], [1.0, 2.0]],
        [[float("nan")], [1.0]],
        [[True], [False]],
        [["bad"], [1.0]],
    ],
)
def test_invalid_vectors_are_not_written(
    vectors: list[list[object]], collection: Collection, embeddings: Mock
) -> None:
    embeddings.embed_documents.side_effect = None
    embeddings.embed_documents.return_value = vectors
    with pytest.raises(ValueError, match="Ollama"):
        index_document(
            "x" * 600,
            source="demo.txt",
            source_id="demo",
            collection=collection,
            embeddings=embeddings,
        )
    assert collection.count() == 0


def test_same_filename_in_different_directories_has_separate_identity(
    tmp_path: Path, collection: Collection, embeddings: Mock
) -> None:
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "same.txt").write_text(directory_name, encoding="utf-8")
        ingest_directory(directory, collection=collection, embeddings=embeddings)
    saved = collection.get()
    assert collection.count() == 2
    assert len({metadata["source_id"] for metadata in saved["metadatas"]}) == 2


def test_persistent_data_after_client_close_and_reopen(tmp_path: Path, embeddings: Mock) -> None:
    settings = Settings(chroma_persist_dir=tmp_path / "persistent")
    with open_collection(settings) as collection:
        index_document(
            "Сохраняемый текст",
            source="persistent.md",
            source_id="persistent",
            collection=collection,
            embeddings=embeddings,
        )
        before = collection.get(include=["documents", "metadatas", "embeddings"])
    with open_collection(settings) as reopened:
        after = reopened.get(include=["documents", "metadatas", "embeddings"])
        for key in ("ids", "documents", "metadatas"):
            assert after[key] == before[key]
        assert after["embeddings"].tolist() == before["embeddings"].tolist()


def test_different_embedding_model_cannot_mix_vectors(tmp_path: Path) -> None:
    path = tmp_path / "persistent"
    with open_collection(Settings(chroma_persist_dir=path)):
        pass
    with (
        pytest.raises(ValueError, match="другую embedding-модель"),
        open_collection(Settings(chroma_persist_dir=path, ollama_embedding_model="another-model")),
    ):
        pass


def test_script_reports_partial_read_failure(
    tmp_path: Path,
    embeddings: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "valid.txt").write_text("Документ", encoding="utf-8")
    (tmp_path / "invalid.md").write_bytes(b"\xff")
    settings = Settings(documents_dir=tmp_path, chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(ingest, "Settings", lambda: settings)
    monkeypatch.setattr(
        ingest, "OllamaResources", lambda settings: nullcontext(Mock(embeddings=embeddings))
    )
    assert ingest.main() == 1
    output = capsys.readouterr().out
    assert '"documents": 1' in output
    assert '"chunks": 1' in output
    assert "invalid.md" in output


def test_script_reports_ollama_failure(
    tmp_path: Path,
    embeddings: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "valid.txt").write_text("Документ", encoding="utf-8")
    settings = Settings(documents_dir=tmp_path, chroma_persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(ingest, "Settings", lambda: settings)
    monkeypatch.setattr(
        ingest, "OllamaResources", lambda settings: nullcontext(Mock(embeddings=embeddings))
    )
    embeddings.embed_documents.side_effect = ConnectionError("Ollama unavailable")
    assert ingest.main() == 1
    assert "Индексация не завершена" in capsys.readouterr().err
