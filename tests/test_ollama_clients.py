import json

import httpx
import pytest
from ollama import AsyncClient, Client

from app.config import Settings
from app.ollama import OllamaResources, UnsafeModelError
from test_ollama import OllamaCheckError, generate


@pytest.fixture
def traffic(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, list, dict]:
    metadata = {
        name: {"details": {"format": "gguf"}, "model_info": {"general.architecture": "local"}}
        for name in ("qwen2.5:1.5b", "nomic-embed-text")
    }
    requests = []
    closed = {"sync": [], "async": []}
    sync_close, async_close = Client.close, AsyncClient.close

    def close_sync(client: Client) -> None:
        closed["sync"].append(id(client))
        sync_close(client)

    async def close_async(client: AsyncClient) -> None:
        closed["async"].append(id(client))
        await async_close(client)

    def handle(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        path = request.url.path
        requests.append((path, payload))
        if path == "/api/show":
            assert set(payload) == {"model"}
            result = metadata[payload["model"]]
        elif path == "/api/embed":
            result = {
                "model": payload["model"],
                "embeddings": [[1.0, 2.0] for _ in payload["input"]],
            }
        elif path == "/api/chat":
            result = {
                "model": payload["model"],
                "message": {"role": "assistant", "content": "Локальный ответ"},
                "done": True,
                "done_reason": "stop",
            }
        else:
            raise AssertionError(f"Unexpected request: {path}")
        return httpx.Response(200, json=result)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(Client, "close", close_sync)
    monkeypatch.setattr(AsyncClient, "close", close_async)
    return metadata, requests, closed


@pytest.mark.parametrize(
    ("model", "remote_field"),
    [("qwen2.5:1.5b", "remote_host"), ("nomic-embed-text", "remote_model")],
)
def test_remote_alias_blocks_documents_before_transport(
    traffic: tuple[dict, list, dict], model: str, remote_field: str
) -> None:
    metadata, requests, closed = traffic
    # The alias still has local-looking details/model_info: remote fields must win.
    metadata[model][remote_field] = "remote-backend"
    with OllamaResources(Settings()) as resources:
        with pytest.raises(UnsafeModelError):
            resources.embeddings.embed_documents(["SECRET DOCUMENT"])
    assert requests and all(path == "/api/show" for path, _ in requests)
    assert "SECRET DOCUMENT" not in json.dumps(requests)
    assert len(closed["sync"]) == len(closed["async"]) == 1


@pytest.mark.parametrize(
    "unknown",
    [
        {},
        {"details": {"format": "gguf"}, "model_info": {}},
        {"details": {"format": "gguf"}, "model_info": {"general.architecture": 42}},
        {
            "details": {"format": "gguf"},
            "model_info": {"general.architecture": "local"},
            "remote_host": 0,
        },
    ],
)
def test_unverified_metadata_fails_closed(traffic: tuple[dict, list, dict], unknown: dict) -> None:
    metadata, requests, _ = traffic
    metadata["qwen2.5:1.5b"] = unknown
    with OllamaResources(Settings()) as resources:
        with pytest.raises(UnsafeModelError):
            resources.chat.invoke("SECRET QUESTION", stream=False)
    assert all(path == "/api/show" for path, _ in requests)


def test_alias_is_rechecked_before_every_text_request(traffic: tuple[dict, list, dict]) -> None:
    metadata, requests, _ = traffic
    with OllamaResources(Settings()) as resources:
        assert resources.embeddings.embed_query("Allowed local input") == [1.0, 2.0]
        metadata["qwen2.5:1.5b"]["remote_host"] = "https://ollama.com"
        with pytest.raises(UnsafeModelError):
            resources.chat.invoke("SECRET QUESTION", stream=False)
    assert not any(path == "/api/chat" for path, _ in requests)
    assert "SECRET QUESTION" not in json.dumps(requests)


def test_reused_clients_close_through_public_sdk_methods(traffic: tuple[dict, list, dict]) -> None:
    _, requests, closed = traffic
    resources = OllamaResources(Settings())
    with resources:
        embeddings, chat = resources.embeddings, resources.chat
        assert chat.temperature == 0.2
        assert chat.keep_alive == embeddings.keep_alive == 0
        assert chat.client_kwargs["trust_env"] is False
        assert chat.client_kwargs["follow_redirects"] is False
        for _ in range(3):
            assert resources.embeddings is embeddings and resources.chat is chat
            assert embeddings.embed_query("Вопрос") == [1.0, 2.0]
            assert chat.invoke("Вопрос", stream=False).text == "Локальный ответ"
    resources.close()  # Normal shutdown and repeated cleanup are safe.
    assert resources._checker.is_closed
    assert len(set(closed["sync"])) == len(closed["sync"]) == 2
    assert len(set(closed["async"])) == len(closed["async"]) == 2
    assert sum(path == "/api/show" for path, _ in requests) == 12


def test_clients_are_closed_on_operation_exception(traffic: tuple[dict, list, dict]) -> None:
    _, _, closed = traffic
    resources = OllamaResources(Settings())
    with pytest.raises(RuntimeError, match="simulated failure"), resources:
        assert resources.chat.keep_alive == 0
        resources.embeddings.embed_query("Вопрос")
        raise RuntimeError("simulated failure")
    assert resources._checker.is_closed
    assert len(closed["sync"]) == len(closed["async"]) == 2


def test_bootstrap_cannot_send_text_to_remote_alias(traffic: tuple[dict, list, dict]) -> None:
    metadata, requests, _ = traffic
    metadata["qwen2.5:1.5b"]["remote_host"] = "remote-host"
    with httpx.Client(trust_env=False) as client, pytest.raises(OllamaCheckError):
        generate(Settings(), client)
    assert all(path == "/api/show" for path, _ in requests)
