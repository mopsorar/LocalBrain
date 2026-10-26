"""Shared settings for the local development environment."""

from ipaddress import ip_address
from pathlib import Path

from pydantic import AnyHttpUrl, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:1.5b", min_length=1)
    ollama_embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    ollama_connect_timeout: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    ollama_read_timeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    documents_dir: Path = Path("knowledge_base")
    chroma_persist_dir: Path = Path("data/chroma")
    uploads_dir: Path = Path("documents/uploads")
    api_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000")
    api_connect_timeout: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    api_read_timeout: float = Field(default=300.0, gt=0, allow_inf_nan=False)

    @field_validator("ollama_model", "ollama_embedding_model")
    @classmethod
    def reject_cloud_model_name(cls, value: str) -> str:
        if value.lower().endswith((":cloud", "-cloud")):
            raise ValueError("Облачные модели Ollama запрещены")
        return value

    @field_validator("ollama_base_url", "api_base_url")
    @classmethod
    def validate_local_origin(cls, value: AnyHttpUrl, info: ValidationInfo) -> AnyHttpUrl:
        if (
            value.username
            or value.password
            or value.query
            or value.fragment
            or value.path not in (None, "/")
        ):
            raise ValueError(
                "Укажите адрес локального сервиса без пути, параметров и учётных данных"
            )
        host = (value.host or "").strip("[]").lower()
        # Compose uses this fixed internal service name; arbitrary DNS hosts stay rejected.
        allowed_hosts = {"localhost"}
        if info.field_name == "ollama_base_url":
            allowed_hosts.add("ollama")
        if host not in allowed_hosts:
            try:
                address = ip_address(host)
            except ValueError as exc:
                raise ValueError("Используйте localhost или локальный IP-адрес сервиса") from exc
            if (
                address.is_unspecified
                or address.is_multicast
                or not (address.is_loopback or address.is_private)
            ):
                raise ValueError("Сервис должен быть доступен по локальному IP-адресу")
        return value

    @field_validator("documents_dir", "chroma_persist_dir", "uploads_dir", mode="before")
    @classmethod
    def resolve_data_path(cls, value: str | Path) -> Path:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Путь к данным не должен быть пустым")
        path = Path(value)
        return (PROJECT_ROOT / path).resolve()

    @model_validator(mode="after")
    def separate_uploads(self) -> "Settings":
        if self.uploads_dir.is_relative_to(self.documents_dir):
            raise ValueError(
                "UPLOADS_DIR должен находиться вне каталога демонстрационных документов"
            )
        return self
