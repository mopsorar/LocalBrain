"""Local-model checks and ownership of LangChain's Ollama HTTP clients."""

import asyncio
import json
from contextlib import ExitStack
from types import TracebackType

import anyio
import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import AsyncClient

from app.config import Settings


class UnsafeModelError(RuntimeError):
    """A model could not be confirmed as local; no input may be sent to it."""


def validate_local_models(settings: Settings, client: httpx.Client) -> None:
    """Only names are sent to /api/show; the SDK ShowResponse drops remote fields."""
    for model in dict.fromkeys((settings.ollama_model, settings.ollama_embedding_model)):
        if model.lower().endswith((":cloud", "-cloud")):
            raise UnsafeModelError("Облачные модели Ollama запрещены.")
        response = client.post(
            f"{str(settings.ollama_base_url).rstrip('/')}/api/show", json={"model": model}
        )
        response.raise_for_status()
        try:
            metadata = response.json()
        except ValueError as exc:
            raise UnsafeModelError("Не удалось проверить локальность модели Ollama.") from exc
        if not isinstance(metadata, dict):
            raise UnsafeModelError("Некорректные метаданные модели Ollama.")
        details = metadata.get("details")
        model_info = metadata.get("model_info")
        if (
            metadata.get("remote_host") not in (None, "")
            or metadata.get("remote_model") not in (None, "")
            or not isinstance(details, dict)
            or details.get("format") != "gguf"
            or not isinstance(model_info, dict)
            or not isinstance(model_info.get("general.architecture"), str)
            or not model_info["general.architecture"].strip()
        ):
            raise UnsafeModelError(
                f"Модель '{model}' не подтверждена как локальная GGUF-модель. "
                "Отправка вопросов и документов запрещена."
            )


class OllamaResources:
    """Own lazy sync/async SDK clients; use synchronously in a worker thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stack = ExitStack()
        self._checker = self._stack.enter_context(httpx.Client(**self._http_options()))
        self._embeddings: OllamaEmbeddings | None = None
        self._chat: ChatOllama | None = None

    def _http_options(self) -> dict:
        return {
            "timeout": httpx.Timeout(
                self.settings.ollama_read_timeout, connect=self.settings.ollama_connect_timeout
            ),
            "trust_env": False,
            "follow_redirects": False,
        }

    def _guard(self, request: httpx.Request) -> None:
        if request.method == "POST" and request.url.path in {
            "/api/embed",
            "/api/embeddings",
            "/api/chat",
            "/api/generate",
        }:
            payload = json.loads(request.content)
            if payload.get("model") not in {
                self.settings.ollama_model,
                self.settings.ollama_embedding_model,
            }:
                raise UnsafeModelError("Запрос использует непроверенную модель Ollama.")
            # Recheck both models on every input request, not just once at startup:
            # a local-looking alias may have been replaced with a remote model.
            validate_local_models(self.settings, self._checker)

    async def _async_guard(self, request: httpx.Request) -> None:
        await anyio.to_thread.run_sync(self._guard, request)

    def _options(self) -> dict:
        return {
            "base_url": str(self.settings.ollama_base_url),
            "keep_alive": 0,
            "num_ctx": 2048,
            "client_kwargs": self._http_options(),
            "sync_client_kwargs": {"event_hooks": {"request": [self._guard]}},
            "async_client_kwargs": {"event_hooks": {"request": [self._async_guard]}},
        }

    def _own(self, component: OllamaEmbeddings | ChatOllama) -> None:
        # langchain-ollama 1.1.0 has no public close/injection API. Access only
        # its two SDK handles here, then use the SDK's public close methods.
        # Never reach through to the SDK's private httpx clients.
        self._stack.callback(component._client.close)
        self._stack.callback(self._close_async, component._async_client)

    @staticmethod
    def _close_async(client: AsyncClient) -> None:
        asyncio.run(client.close())

    @property
    def embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.settings.ollama_embedding_model, **self._options()
            )
            self._own(self._embeddings)
        return self._embeddings

    @property
    def chat(self) -> ChatOllama:
        if self._chat is None:
            self._chat = ChatOllama(
                model=self.settings.ollama_model,
                temperature=0.2,
                num_predict=256,
                **self._options(),
            )
            self._own(self._chat)
        return self._chat

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "OllamaResources":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
