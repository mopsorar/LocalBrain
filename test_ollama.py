"""Run a single local Ollama generation check: python test_ollama.py."""

import json
import sys
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.ollama import UnsafeModelError, validate_local_models


class OllamaCheckError(RuntimeError):
    """An expected failure of the local Ollama check."""


def generate(settings: Settings, client: httpx.Client) -> dict[str, Any]:
    """Request and validate a complete response without downloading models."""
    url = f"{str(settings.ollama_base_url).rstrip('/')}/api/generate"
    try:
        validate_local_models(settings, client)
        response = client.post(
            url,
            json={
                "model": settings.ollama_model,
                "prompt": "Привет! Коротко подтверди, что локальная генерация работает.",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=httpx.Timeout(
                settings.ollama_read_timeout,
                connect=settings.ollama_connect_timeout,
            ),
        )
    except httpx.TimeoutException as exc:
        raise OllamaCheckError(
            "Истёк тайм-аут запроса к Ollama. Проверьте сервис и ресурсы компьютера; "
            "при необходимости увеличьте OLLAMA_READ_TIMEOUT в .env."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaCheckError(
            f"Не удалось подключиться к Ollama ({settings.ollama_base_url}). "
            "Проверьте адрес и запущен ли локальный сервис (ollama serve)."
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            model = json.loads(exc.request.content).get("model", "")
            raise OllamaCheckError(f"Модель '{model}' не найдена. Проверьте ollama list.") from exc
        raise OllamaCheckError("Не удалось проверить метаданные локальных моделей Ollama.") from exc
    except UnsafeModelError as exc:
        raise OllamaCheckError("Не удалось подтвердить локальность обеих моделей Ollama.") from exc

    if not response.is_success:
        try:
            body = response.json()
        except ValueError:
            body = None
        detail = body.get("error") if isinstance(body, dict) else None
        detail = str(detail or response.text or response.reason_phrase)[:500]
        if (
            response.status_code == 404
            and "model" in detail.lower()
            and "not found" in detail.lower()
        ):
            raise OllamaCheckError(
                f"Модель '{settings.ollama_model}' отсутствует в локальной Ollama. "
                "Проверьте ollama list и OLLAMA_MODEL в .env. "
                "Автоматическая загрузка моделей не выполняется."
            )
        raise OllamaCheckError(f"Ollama вернула HTTP {response.status_code}: {detail}")

    try:
        result = response.json()
    except ValueError as exc:
        raise OllamaCheckError("Ollama вернула некорректный JSON вместо полного ответа.") from exc
    if not isinstance(result, dict):
        raise OllamaCheckError("Некорректный ответ Ollama: ожидался JSON-объект.")
    if result.get("error"):
        raise OllamaCheckError(f"Ollama сообщила об ошибке: {str(result['error'])[:500]}")
    text = result.get("response")
    if not isinstance(text, str) or not text.strip() or result.get("done") is not True:
        raise OllamaCheckError(
            "Некорректный ответ Ollama: ожидались непустая строка 'response' и 'done: true'."
        )
    return result


def main() -> int:
    try:
        settings = Settings()
        # Ignore system proxy variables and never follow redirects to another service.
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            result = generate(settings, client)
    except ValidationError as exc:
        print(f"Ошибка конфигурации LocalBrain: {exc}", file=sys.stderr)
        return 1
    except OllamaCheckError as exc:
        print(f"Проверка Ollama не пройдена: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
