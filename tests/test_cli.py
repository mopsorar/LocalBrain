from unittest.mock import Mock

import httpx
import pytest

import main as cli
from app.config import Settings


def test_cli_receives_multiple_answers_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = iter(["Первый вопрос", "Второй вопрос", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    monkeypatch.setenv("API_BASE_URL", "http://localhost:9876")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:9876/chat"
        requests.append(request)
        return httpx.Response(200, json={"answer": f"Ответ {len(requests)}"})

    real_client = httpx.Client

    def make_client(**kwargs: object) -> httpx.Client:
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(cli.httpx, "Client", make_client)
    assert cli.main() == 0
    output = capsys.readouterr()
    assert "Ответ 1" in output.out and "Ответ 2" in output.out
    assert len(requests) == 2 and not output.err


def test_connection_failure_is_explained_and_loop_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = iter(["Вопрос", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.post.side_effect = httpx.ConnectError("offline")
    monkeypatch.setattr(cli.httpx, "Client", Mock(return_value=client))
    assert cli.main() == 0
    assert "API недоступен" in capsys.readouterr().err
    client.post.assert_called_once()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503), "HTTP 503"),
        (httpx.Response(200, text="invalid JSON"), "некорректный JSON"),
        (httpx.Response(200, json={"answer": " "}), "Некорректный ответ"),
    ],
)
def test_api_errors(response: httpx.Response, message: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as client:
        with pytest.raises(cli.APIError, match=message):
            cli.ask_api("Вопрос", client, Settings())


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cli_exits_on_eof_or_ctrl_c(
    error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", Mock(side_effect=error))
    assert cli.main() == 0
