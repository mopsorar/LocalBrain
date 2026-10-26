from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


def test_defaults_are_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.ollama_model == "qwen2.5:1.5b"
    assert settings.ollama_embedding_model == "nomic-embed-text"
    assert settings.documents_dir == PROJECT_ROOT / "knowledge_base"
    assert settings.chroma_persist_dir == PROJECT_ROOT / "data/chroma"
    assert settings.uploads_dir == PROJECT_ROOT / "documents/uploads"
    assert str(settings.api_base_url) == "http://127.0.0.1:8000/"


def test_dotenv_and_environment_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_MODEL=local-model:latest\n"
        "OLLAMA_EMBEDDING_MODEL=local-embedding:latest\n"
        "DOCUMENTS_DIR=данные\n"
        "CHROMA_PERSIST_DIR=storage/chroma\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_MODEL", "environment-model:latest")
    settings = Settings(_env_file=env_file)
    assert settings.ollama_model == "environment-model:latest"
    assert settings.ollama_embedding_model == "local-embedding:latest"
    assert settings.documents_dir == PROJECT_ROOT / "данные"
    assert settings.chroma_persist_dir == PROJECT_ROOT / "storage/chroma"


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:11434", "http://[::1]:11434", "http://192.168.1.5:11434"]
)
def test_local_addresses_are_supported(url: str) -> None:
    assert Settings(ollama_base_url=url).ollama_base_url.port == 11434


def test_compose_service_name_is_allowed_only_for_ollama() -> None:
    settings = Settings(ollama_base_url="http://ollama:11434")
    assert settings.ollama_base_url.host == "ollama"
    with pytest.raises(ValidationError):
        Settings(api_base_url="http://ollama:8000")
    with pytest.raises(ValidationError):
        Settings(ollama_base_url="http://arbitrary-host:11434")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ollama_base_url", "invalid-url"),
        ("ollama_base_url", "https://ollama.com"),
        ("ollama_base_url", "http://8.8.8.8"),
        ("ollama_base_url", "http://localhost:11434/api"),
        ("ollama_model", "   "),
        ("ollama_embedding_model", ""),
        ("ollama_connect_timeout", 0),
        ("ollama_read_timeout", -1),
        ("ollama_read_timeout", "nan"),
        ("documents_dir", ""),
        ("api_base_url", "https://example.com"),
        ("ollama_model", "qwen:cloud"),
        ("ollama_embedding_model", "embedding:latest-cloud"),
    ],
)
def test_invalid_settings_are_rejected(name: str, value: str | int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{name: value})


def test_uploads_cannot_share_demo_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="UPLOADS_DIR"):
        Settings(documents_dir=tmp_path, uploads_dir=tmp_path / "uploads")
