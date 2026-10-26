import json
from typing import Any

import httpx
import pytest

import test_ollama as probe
from app.config import Settings


@pytest.fixture(autouse=True)
def local_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport

    def make_transport(handler: Any) -> httpx.MockTransport:
        def route(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/show":
                assert set(json.loads(request.content)) == {"model"}
                return httpx.Response(
                    200,
                    json={
                        "details": {"format": "gguf"},
                        "model_info": {"general.architecture": "test"},
                    },
                )
            return handler(request)

        return transport(route)

    monkeypatch.setattr(httpx, "MockTransport", make_transport)


def test_generate_uses_configured_request_and_preserves_full_response() -> None:
    settings = Settings(
        ollama_base_url="http://127.0.0.1:12345",
        ollama_model="local-test:latest",
        ollama_connect_timeout=2,
        ollama_read_timeout=42,
    )
    payload = {"response": "Ответ " * 1000, "done": True, "eval_count": 1000}
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "http://127.0.0.1:12345/api/generate"
        body = json.loads(request.content)
        assert body["model"] == settings.ollama_model
        assert body["stream"] is False
        assert body["keep_alive"] == 0
        assert body["prompt"]
        assert request.extensions["timeout"]["connect"] == 2
        assert request.extensions["timeout"]["read"] == 42
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert probe.generate(settings, client) == payload
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError, "Не удалось подключиться"),
        (httpx.ConnectTimeout, "тайм-аут"),
        (httpx.ReadTimeout, "тайм-аут"),
    ],
)
def test_network_errors_are_explained(error: type[httpx.RequestError], message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated failure", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(probe.OllamaCheckError, match=message),
    ):
        probe.generate(Settings(), client)


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (404, {"error": "model 'qwen2.5:1.5b' not found"}, "Модель .* отсутствует"),
        (404, {"error": "route not found"}, "HTTP 404"),
        (500, {"error": "insufficient memory"}, "HTTP 500: insufficient memory"),
        (200, {"error": "generation failed"}, "Ollama сообщила об ошибке"),
        (200, [], "JSON-объект"),
        (200, {"done": True}, "Некорректный ответ"),
        (200, {"response": 123, "done": True}, "Некорректный ответ"),
        (200, {"response": "   ", "done": True}, "Некорректный ответ"),
        (200, {"response": "partial", "done": False}, "Некорректный ответ"),
        (200, {"response": "text", "done": 1}, "Некорректный ответ"),
    ],
)
def test_server_errors_and_invalid_payloads(status: int, payload: Any, message: str) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=payload))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(probe.OllamaCheckError, match=message),
    ):
        probe.generate(Settings(), client)


@pytest.mark.parametrize(
    ("status", "text", "message"),
    [(200, "not JSON", "некорректный JSON"), (503, "unavailable", "HTTP 503: unavailable")],
)
def test_non_json_responses(status: int, text: str, message: str) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=text))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(probe.OllamaCheckError, match=message),
    ):
        probe.generate(Settings(), client)


@pytest.mark.parametrize("status", [200, 307])
def test_main_output_and_no_redirects(
    status: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"response": "Полный ответ " * 1000, "done": True, "total_duration": 123}
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=payload, headers={"Location": "https://example.com"})

    real_client = httpx.Client

    def create_client(**kwargs: Any) -> httpx.Client:
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(probe.httpx, "Client", create_client)
    assert probe.main() == (0 if status == 200 else 1)
    captured = capsys.readouterr()
    assert len(requests) == 1
    if status == 200:
        assert json.loads(captured.out) == payload
        assert not captured.err
    else:
        assert not captured.out
        assert "HTTP 307" in captured.err


def test_main_reports_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OLLAMA_READ_TIMEOUT", "0")
    assert probe.main() == 1
    assert "Ошибка конфигурации" in capsys.readouterr().err
