import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langsmith.utils import tracing_is_enabled

import app.main as api_module
from app.config import Settings
from app.ingestion import index_document, open_collection
from rag_bot import OllamaUnavailableError, RAGBot


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    settings = Settings(
        documents_dir=tmp_path / "demos",
        uploads_dir=tmp_path / "uploads",
        chroma_persist_dir=tmp_path / "chroma",
    )
    bot = Mock(spec=RAGBot)
    bot.ask.return_value = "Текст ответа"

    @contextmanager
    def mocked_collection(settings: Settings) -> Iterator[Mock]:
        yield Mock()

    monkeypatch.setattr(api_module, "Settings", lambda: settings)
    monkeypatch.setattr(api_module, "RAGBot", lambda settings, **kwargs: bot)
    monkeypatch.setattr(api_module, "open_collection", mocked_collection)
    monkeypatch.setattr(api_module, "OllamaResources", Mock(return_value=Mock()))
    monkeypatch.setattr(api_module, "index_document", Mock(return_value=2))
    with TestClient(api_module.app) as client:
        yield client


def test_chat_success_and_empty_collection(client: TestClient) -> None:
    assert client.post("/chat", json={"question": "  Вопрос  "}).json() == {
        "answer": "Текст ответа"
    }
    client.app.state.bot.ask.assert_called_once_with("Вопрос")
    client.app.state.bot.ask.return_value = "Я пока ничего не знаю"
    response = client.post("/chat", json={"question": "Что ты знаешь?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Я пока ничего не знаю"}


def test_lifespan_closes_resources_on_startup_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    resources = Mock()
    monkeypatch.setattr(api_module, "OllamaResources", Mock(return_value=resources))
    monkeypatch.setattr(api_module, "RAGBot", Mock(side_effect=RuntimeError("startup failure")))
    with pytest.raises(RuntimeError, match="startup failure"), TestClient(api_module.app):
        pass
    resources.close.assert_called_once()


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": " \n\t"}, {}])
def test_chat_validation(client: TestClient, payload: dict[str, str]) -> None:
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    client.app.state.bot.ask.assert_not_called()


def test_unavailable_ollama_and_private_internal_errors(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    client.app.state.bot.ask.side_effect = OllamaUnavailableError("private details")
    assert client.post("/chat", json={"question": "secret question"}).status_code == 503
    client.app.state.bot.ask.side_effect = RuntimeError("private details")
    response = client.post("/chat", json={"question": "secret question"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Внутренняя ошибка сервера."}
    assert "private details" not in caplog.text + response.text
    assert "secret question" not in caplog.text + response.text


def test_upload_storage_and_stable_source_identity(client: TestClient) -> None:
    for text in ("Первый текст", "Обновлённый текст"):
        response = client.post(
            "/ingest", files={"file": ("demo.md", text.encode(), "text/markdown")}
        )
        assert response.status_code == 200
        assert response.json() == {"chunks_added": 2}
    calls = api_module.index_document.call_args_list
    assert calls[0].kwargs["source_id"] == calls[1].kwargs["source_id"]
    assert calls[0].kwargs["source_id"].startswith("upload:")
    assert calls[1].args[0] == "Обновлённый текст"
    assert calls[1].kwargs["source"] == "demo.md"
    files = list(client.app.state.settings.uploads_dir.iterdir())
    assert len(files) == 1
    assert files[0].name != "demo.md"
    assert files[0].read_text(encoding="utf-8") == "Обновлённый текст"


@pytest.mark.parametrize(
    ("filename", "content", "status"),
    [
        ("bad.pdf", b"text", 415),
        ("bad.txt", b"\xff", 400),
        ("empty.md", b" \n ", 422),
        ("../private.md", b"text", 400),
        ("..\\private.md", b"text", 400),
    ],
)
def test_rejected_uploads(client: TestClient, filename: str, content: bytes, status: int) -> None:
    response = client.post("/ingest", files={"file": (filename, content)})
    assert response.status_code == status
    api_module.index_document.assert_not_called()
    assert not client.app.state.settings.uploads_dir.exists()


def test_file_size_and_chunked_body_limits(client: TestClient) -> None:
    response = client.post(
        "/ingest", files={"file": ("large.txt", b"x" * (api_module.MAX_UPLOAD_BYTES + 1))}
    )
    assert response.status_code == 413
    # No Content-Length: reject the body before the multipart parser can spool it to disk.
    response = client.post(
        "/ingest",
        content=iter([b"x" * (api_module.MAX_UPLOAD_BYTES + 64 * 1024 + 1)]),
        headers={"Content-Type": "multipart/form-data; boundary=example"},
    )
    assert response.status_code == 413
    api_module.index_document.assert_not_called()


def test_failed_upload_preserves_previous_file(client: TestClient) -> None:
    assert client.post("/ingest", files={"file": ("same.txt", b"old text")}).status_code == 200
    api_module.index_document.side_effect = ConnectionError("Ollama offline")
    assert client.post("/ingest", files={"file": ("same.txt", b"new text")}).status_code == 503
    files = list(client.app.state.settings.uploads_dir.iterdir())
    assert len(files) == 1
    assert files[0].read_bytes() == b"old text"


def test_uploaded_knowledge_is_immediately_visible_without_restart(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = Mock(spec=Embeddings)
    embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.0, 0.0] for text in texts]
    embeddings.embed_query.return_value = [1.0, 0.0, 0.0]
    monkeypatch.setattr(api_module, "open_collection", open_collection)
    monkeypatch.setattr(api_module, "index_document", index_document)
    model = Mock()

    def answer_from_context(prompt: object, **kwargs: object) -> AIMessage:
        messages = prompt.to_messages()
        context = json.loads(messages[1].content.split("\n", 1)[1])
        assert any(item["text"] == "Код проекта Лира — зелёный маяк." for item in context)
        return AIMessage(content="Код проекта Лира — зелёный маяк.")

    model.invoke.side_effect = answer_from_context
    resources = client.app.state.ollama
    resources.embeddings = embeddings
    resources.chat = model
    settings = client.app.state.settings
    bot = RAGBot(settings, ollama=resources)
    client.app.state.bot = bot
    for _ in range(2):
        response = client.post(
            "/ingest", files={"file": ("lira.txt", "Код проекта Лира — зелёный маяк.".encode())}
        )
        assert response.json() == {"chunks_added": 1}
    response = client.post("/chat", json={"question": "Какой код проекта Лира?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Код проекта Лира — зелёный маяк."}
    assert client.app.state.bot is bot
    with open_collection(settings) as collection:
        assert collection.count() == 1
    assert embeddings.embed_documents.call_count == 1


def test_offload_shared_concurrency_limit_and_no_cloud_tracing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum = 0
    thread_ids = []
    tracing = []
    loop_id = None

    def heavy(result: object) -> object:
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        thread_ids.append(threading.get_ident())
        tracing.append(tracing_is_enabled())
        entered.set()
        assert release.wait(timeout=3)
        with state_lock:
            active -= 1
        return result

    client.app.state.bot.ask.side_effect = lambda question: heavy("Ответ")
    api_module.index_document.side_effect = lambda *args, **kwargs: heavy(1)

    async def scenario() -> None:
        nonlocal loop_id
        loop_id = threading.get_ident()
        # Use one event loop for the limiter and both concurrent ASGI requests.
        client.app.state.operation_limiter = api_module.anyio.CapacityLimiter(1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app), base_url="http://testserver"
        ) as http:
            first = asyncio.create_task(http.post("/chat", json={"question": "Вопрос"}))
            try:
                async with asyncio.timeout(2):
                    while not entered.is_set():
                        await asyncio.sleep(0.005)
                second = asyncio.create_task(
                    http.post("/ingest", files={"file": ("demo.txt", b"text")})
                )
                await asyncio.sleep(0.02)
                assert not first.done()  # Event loop advanced while the sync worker was blocked.
                assert not second.done()
            finally:
                release.set()
            responses = await asyncio.gather(first, second)
            assert [response.status_code for response in responses] == [200, 200]

    asyncio.run(scenario())
    assert maximum == 1
    assert len(thread_ids) == 2 and all(thread_id != loop_id for thread_id in thread_ids)
    assert tracing == [False, False]
