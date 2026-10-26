"""Interactive HTTP client. Start the API separately before running python main.py."""

import sys

import httpx
from pydantic import ValidationError

from app.config import Settings


class APIError(RuntimeError):
    """An expected failure of the local API request."""


def ask_api(question: str, client: httpx.Client, settings: Settings) -> str:
    url = f"{str(settings.api_base_url).rstrip('/')}/chat"
    try:
        response = client.post(url, json={"question": question})
    except httpx.TimeoutException as exc:
        raise APIError(
            "Истёк тайм-аут API. Возможно, предыдущая операция ещё выполняется."
        ) from exc
    except httpx.RequestError as exc:
        raise APIError("API недоступен. Проверьте API_BASE_URL и запущен ли Uvicorn.") from exc
    if not response.is_success:
        raise APIError(f"API вернул HTTP {response.status_code}.")
    try:
        body = response.json()
    except ValueError as exc:
        raise APIError("API вернул некорректный JSON.") from exc
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("answer"), str)
        or not body["answer"].strip()
    ):
        raise APIError("Некорректный ответ API: ожидалось непустое текстовое поле 'answer'.")
    return body["answer"].strip()


def main() -> int:
    try:
        settings = Settings()
        with httpx.Client(
            timeout=httpx.Timeout(settings.api_read_timeout, connect=settings.api_connect_timeout),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            print("LocalBrain. Введите вопрос или exit для выхода.")
            while True:
                question = input("Вы> ").strip()
                if question.casefold() == "exit":
                    return 0
                if not question:
                    print("Введите непустой вопрос.")
                    continue
                try:
                    print(ask_api(question, client, settings))
                except APIError as exc:
                    print(str(exc), file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print("\nЗавершение работы.")
        return 0
    except ValidationError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
