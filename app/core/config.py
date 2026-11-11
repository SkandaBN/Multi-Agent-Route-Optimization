"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Smart Logistics application."""

    app_name: str = "Smart Logistics API"
    app_env: str = "development"
    app_version: str = "0.1.0"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"
    crewai_telemetry_opt_out: bool = True

    ortools_solver_time_limit_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

