import pytest

import langroid.language_models.openai_gpt as lm
from langroid.utils.configuration import settings


def _clear_vertexai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chat_model", "")
    for name in (
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "GEMINI_API_BASE",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_vertexai_prefix_builds_regional_endpoint(monkeypatch):
    _clear_vertexai_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    provider = lambda: "token"  # noqa: E731

    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_key_provider=provider,
        )
    )

    assert llm.is_vertexai
    assert llm.is_gemini
    assert llm.config.chat_model == "google/gemini-3-flash"
    assert llm.api_base == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "projects/test-project/locations/us-central1/endpoints/openapi"
    )
    assert llm.config.api_key_provider is provider


def test_vertexai_global_location_uses_global_host(monkeypatch):
    _clear_vertexai_env(monkeypatch)
    provider = lambda: "token"  # noqa: E731

    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            vertexai_project_id="test-project",
            vertexai_location="global",
            api_key_provider=provider,
        )
    )

    assert llm.api_base == (
        "https://aiplatform.googleapis.com/v1beta1/"
        "projects/test-project/locations/global/endpoints/openapi"
    )


def test_vertexai_explicit_api_base_takes_precedence(monkeypatch):
    _clear_vertexai_env(monkeypatch)
    custom_base = "https://vertex.example/v1"

    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base=custom_base,
            api_key_provider=lambda: "token",
        )
    )

    assert llm.api_base == custom_base
    assert llm.config.chat_model == "google/gemini-3-flash"


@pytest.mark.parametrize(
    ("project_id", "location", "message"),
    [
        ("", "us-central1", "Google Cloud project"),
        ("test-project", "", "Google Cloud location"),
    ],
)
def test_vertexai_requires_endpoint_components(
    monkeypatch, project_id, location, message
):
    _clear_vertexai_env(monkeypatch)

    with pytest.raises(ValueError, match=message):
        lm.OpenAIGPT(
            lm.OpenAIGPTConfig(
                chat_model="vertexai/google/gemini-3-flash",
                vertexai_project_id=project_id,
                vertexai_location=location,
                api_key_provider=lambda: "token",
            )
        )


@pytest.mark.parametrize(
    "chat_model",
    ["vertexai/", "vertexai/gemini-3-flash", "vertexai//gemini-3-flash"],
)
def test_vertexai_requires_publisher_qualified_model(monkeypatch, chat_model):
    _clear_vertexai_env(monkeypatch)

    with pytest.raises(ValueError, match="vertexai/<publisher>/<model>"):
        lm.OpenAIGPT(
            lm.OpenAIGPTConfig(
                chat_model=chat_model,
                api_base="https://vertex.example/v1",
                api_key_provider=lambda: "token",
            )
        )


def test_vertexai_creates_adc_provider_when_auth_is_not_supplied(monkeypatch):
    _clear_vertexai_env(monkeypatch)
    provider = lambda: "adc-token"  # noqa: E731
    monkeypatch.setattr(lm, "_create_vertexai_token_provider", lambda: provider)

    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base="https://vertex.example/v1",
        )
    )

    assert llm.config.api_key_provider is provider


def test_vertexai_preserves_explicit_static_api_key(monkeypatch):
    _clear_vertexai_env(monkeypatch)

    def unexpected_provider():
        raise AssertionError("ADC provider should not be created")

    monkeypatch.setattr(lm, "_create_vertexai_token_provider", unexpected_provider)
    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base="https://vertex.example/v1",
            api_key="explicit-token",
        )
    )

    assert llm.api_key == "explicit-token"
    assert llm.config.api_key_provider is None
