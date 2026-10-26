"""Local HTTP API. Run Uvicorn with one worker to share the operation limiter."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

import anyio
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langsmith import tracing_context
from ollama import ResponseError
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings
from app.ingestion import index_document, open_collection
from app.ollama import OllamaResources, UnsafeModelError
from rag_bot import OllamaUnavailableError, RAGBot

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class UploadBodyLimit:
    """Bound the multipart body before parsing/spooling, including chunked requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/ingest" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        limit = MAX_UPLOAD_BYTES + 64 * 1024  # Allow bounded multipart headers/boundaries.
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            part = message.get("body", b"")
            if len(body) + len(part) > limit:
                await JSONResponse(
                    status_code=413, content={"detail": "Запрос загрузки слишком велик."}
                )(scope, receive, send)
                return
            body.extend(part)
            if not message.get("more_body", False):
                break

        async def receive_body() -> Message:
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, receive_body, send)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str = Field(min_length=1)


class IngestResponse(BaseModel):
    chunks_added: int = Field(
        ge=1, description="Количество актуальных чанков загруженного источника"
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Settings reads .env and resolves paths, so initialize it outside the event loop too.
    app.state.settings = await anyio.to_thread.run_sync(Settings)
    app.state.ollama = await anyio.to_thread.run_sync(OllamaResources, app.state.settings)
    try:
        app.state.bot = RAGBot(app.state.settings, ollama=app.state.ollama)
        app.state.operation_limiter = anyio.CapacityLimiter(1)
        yield
    finally:
        await anyio.to_thread.run_sync(app.state.ollama.close)


app = FastAPI(
    title="LocalBrain",
    lifespan=lifespan,
    telemetry={
        "tracing": False,
        "metrics": False,
        "logs": False,
        "operation_spans": False,
        "auto_configure": False,
    },
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(UploadBodyLimit)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Default validation responses can echo the complete input question.
    return JSONResponse(status_code=422, content={"detail": "Некорректные данные запроса."})


def _local_operation[T](function: Callable[..., T], *args: object) -> T:
    with tracing_context(enabled=False):
        return function(*args)


async def _run_operation[T](request: Request, function: Callable[..., T], *args: object) -> T:
    try:
        return await anyio.to_thread.run_sync(
            _local_operation,
            function,
            *args,
            limiter=request.app.state.operation_limiter,
        )
    except HTTPException:
        raise
    except (
        OllamaUnavailableError,
        UnsafeModelError,
        ConnectionError,
        httpx.HTTPError,
        ResponseError,
    ) as exc:
        logger.warning("Локальная Ollama не обслужила запрос (%s)", type(exc).__name__)
        raise HTTPException(
            503, "Локальная Ollama недоступна или нужная модель не готова."
        ) from exc
    except Exception as exc:
        # Do not log exception messages: upstream errors can include private document text.
        logger.error("Ошибка операции %s (%s)", request.url.path, type(exc).__name__)
        raise HTTPException(500, "Внутренняя ошибка сервера.") from exc


def _index_upload(settings: Settings, upload: UploadFile, ollama: OllamaResources) -> int:
    name = upload.filename or ""
    if (
        not name
        or len(name) > 255
        or name != name.strip()
        or any(character in name for character in '/\\:<>"|?*')
        or any(ord(character) < 32 for character in name)
        or name in {".", ".."}
    ):
        raise HTTPException(400, "Небезопасное имя файла.")
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".md"}:
        raise HTTPException(415, "Разрешены только файлы .txt и .md.")
    if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл превышает лимит 2 МиБ.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл превышает лимит 2 МиБ.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, "Файл должен содержать корректный UTF-8.") from exc
    if not text.strip():
        raise HTTPException(422, "Документ не должен быть пустым.")

    # A separate namespace prevents an upload from replacing a demo source.
    identity = sha256(name.casefold().encode()).hexdigest()
    source_id = "upload:" + identity
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.uploads_dir / (identity + suffix)
    temporary = None
    try:
        with NamedTemporaryFile(dir=settings.uploads_dir, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        with open_collection(settings) as collection:
            chunks = index_document(
                text,
                source=name,
                source_id=source_id,
                collection=collection,
                embeddings=ollama.embeddings,
            )
        temporary.replace(destination)
        return chunks
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    answer = await _run_operation(request, request.app.state.bot.ask, body.question)
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(500, "Внутренняя ошибка сервера.")
    return ChatResponse(answer=answer)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, file: Annotated[UploadFile, File()]) -> IngestResponse:
    chunks = await _run_operation(
        request, _index_upload, request.app.state.settings, file, request.app.state.ollama
    )
    if not isinstance(chunks, int) or chunks < 1:
        raise HTTPException(500, "Внутренняя ошибка сервера.")
    return IngestResponse(chunks_added=chunks)
