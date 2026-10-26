"""Index the configured local document directory: python ingest.py."""

import json
import logging
import sys

import httpx
from chromadb.errors import ChromaError
from ollama import ResponseError

from app.config import Settings
from app.ingestion import ingest_directory, open_collection
from app.ollama import OllamaResources, UnsafeModelError


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        settings = Settings()
        if not settings.documents_dir.is_dir():
            raise FileNotFoundError("Каталог документов не найден: проверьте DOCUMENTS_DIR")
        with open_collection(settings) as collection, OllamaResources(settings) as ollama:
            result = ingest_directory(
                settings.documents_dir,
                collection=collection,
                embeddings=ollama.embeddings,
            )
            print(
                json.dumps(
                    {
                        "documents": len(result.files),
                        "chunks": sum(result.files.values()),
                        "files": result.files,
                        "errors": result.errors,
                        "collection_count": collection.count(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 1 if result.errors else 0
    except (
        OSError,
        ValueError,
        httpx.HTTPError,
        ResponseError,
        ChromaError,
        UnsafeModelError,
    ) as exc:
        print(f"Индексация не завершена: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
