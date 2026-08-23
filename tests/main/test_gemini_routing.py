import langroid.language_models as lm
from langroid.utils.configuration import settings


def test_gemini_config_subclass_default_preserves_api_base(monkeypatch):
    """Gemini configs retain an API base defined by a config subclass."""

    class VertexConfig(lm.OpenAIGPTConfig):
        api_base: str | None = "https://vertex.example/v1"

    monkeypatch.setattr(settings, "chat_model", "")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("GEMINI_API_BASE", "https://gemini-env.example/v1")

    llm = lm.OpenAIGPT(VertexConfig(chat_model="google/gemini-2.0-flash"))

    assert llm.is_gemini
    assert llm.api_base == "https://vertex.example/v1"


def test_gemini_ignores_lowercase_openai_api_base(monkeypatch):
    """Gemini routing ignores case-insensitive ambient OpenAI settings."""
    openai_base = "https://ambient-openai.example/v1"
    gemini_base = "https://gemini-env.example/v1"
    monkeypatch.setattr(settings, "chat_model", "")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("openai_api_base", openai_base)
    monkeypatch.setenv("GEMINI_API_BASE", gemini_base)

    config = lm.OpenAIGPTConfig(chat_model="google/gemini-2.0-flash")
    llm = lm.OpenAIGPT(config)

    assert config.api_base == openai_base
    assert llm.api_base == gemini_base
