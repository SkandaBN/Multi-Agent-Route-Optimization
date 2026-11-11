"""Tests for Gemini configuration, LLM factory, and MockLogisticsLLM."""

from crewai.llms.providers.gemini.completion import GeminiCompletion
import pytest

from app.core.config import Settings
from app.core.llm import (
    MockLogisticsLLM,
    create_gemini_llm,
    create_llm,
    normalize_gemini_model_name,
    resolve_gemini_api_key,
)


def test_normalize_gemini_model_name() -> None:
    """Model names normalize cleanly for CrewAI's native Gemini provider."""
    assert normalize_gemini_model_name("gemini-2.5-flash") == "gemini/gemini-2.5-flash"
    assert normalize_gemini_model_name("gemini/gemini-2.5-pro") == "gemini/gemini-2.5-pro"
    assert normalize_gemini_model_name("  ") == "gemini/gemini-2.5-flash"


def test_resolve_gemini_api_key_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMINI_API_KEY and GOOGLE_API_KEY are supported while placeholders are ignored."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    empty_settings = Settings(gemini_api_key="your_gemini_api_key_here", google_api_key="")
    assert resolve_gemini_api_key(empty_settings) == ""

    monkeypatch.setenv("GOOGLE_API_KEY", "google-fallback-key")
    assert resolve_gemini_api_key(empty_settings) == "google-fallback-key"

    monkeypatch.setenv("GEMINI_API_KEY", "primary-gemini-key")
    assert resolve_gemini_api_key(empty_settings) == "primary-gemini-key"


def test_create_gemini_llm_requires_key_and_configures_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_gemini_llm fails cleanly when unset and builds GeminiCompletion when configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    no_key_settings = Settings(gemini_api_key="", google_api_key="")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_gemini_llm(no_key_settings)

    monkeypatch.setenv("GEMINI_API_KEY", "test-offline-gemini-key")
    llm = create_gemini_llm(
        Settings(gemini_api_key="", gemini_model_name="gemini-2.5-flash", gemini_temperature=0.0)
    )
    assert isinstance(llm, GeminiCompletion)
    assert llm.model == "gemini-2.5-flash"
    assert llm.provider == "gemini"


def test_create_llm_falls_back_to_mock_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode selects MockLogisticsLLM deterministically when no Gemini key exists."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(gemini_api_key="", google_api_key="", llm_mode="auto")

    llm = create_llm(settings)
    assert isinstance(llm, MockLogisticsLLM)
    assert llm.supports_function_calling() is False

