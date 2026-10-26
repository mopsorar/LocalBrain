FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/localbrain \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Keep the dependency layer cached when application code changes.
COPY requirements.txt ./
RUN python -m pip install --only-binary=:all: --requirement requirements.txt

RUN groupadd --gid 10001 localbrain \
    && useradd --uid 10001 --gid localbrain --create-home --shell /usr/sbin/nologin localbrain \
    && mkdir -p /app/data/chroma /app/documents/uploads \
    && chown -R localbrain:localbrain /app/data /app/documents

COPY app/ ./app/
COPY rag_bot.py ingest.py main.py test_ollama.py ./
COPY knowledge_base/ ./knowledge_base/

USER 10001:10001
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8"]
