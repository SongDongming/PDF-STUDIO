from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Credentials are accepted only through protected environment variables or a
    future credential provider. They are never returned by the public settings
    API.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "多模态 PDF 知识库"
    env: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:4321"])

    database_url: str = "sqlite+pysqlite:///:memory:"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: SecretStr | None = None
    minio_secret_key: SecretStr | None = None
    minio_secure: bool = False
    minio_bucket: str = "pdfwiki"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None

    ocr_base_url: str | None = None
    agent_runtime: Literal["legacy", "deepagents"] = "deepagents"

    @field_validator("cors_origins")
    @classmethod
    def reject_wildcard_with_credentials(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError("CORS origins must be explicit; wildcard is not allowed")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
