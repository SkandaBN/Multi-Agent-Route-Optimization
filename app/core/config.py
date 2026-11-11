"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
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

    gemini_api_key: str = ""
    google_api_key: str = ""
    gemini_model_name: str = "gemini-2.5-flash"
    gemini_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_mode: Literal["auto", "gemini", "mock"] = "auto"

    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"
    crewai_telemetry_opt_out: bool = True

    ortools_solver_time_limit_seconds: int = Field(default=30, gt=0)
    fuel_efficiency_km_per_liter: float = Field(default=8.0, gt=0.0)
    fuel_price_per_liter: float = Field(default=1.5, ge=0.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolved_gemini_api_key(self) -> str:
        """Return the configured Gemini API key from GEMINI_API_KEY or GOOGLE_API_KEY."""
        return (self.gemini_api_key or self.google_api_key).strip()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

