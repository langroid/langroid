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
        "openai_api_key",
        "GEMINI_API_BASE",
        "GEMINI_API_KEY",
        "VERTEX_API_BASE",
        "VERTEX_API_KEY",
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


def test_vertexai_ignores_inherited_openai_api_key(monkeypatch):
    """An OPENAI_API_KEY in the env must not suppress ADC or reach Google."""
    _clear_vertexai_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated-openai-key")
    provider = lambda: "adc-token"  # noqa: E731
    monkeypatch.setattr(lm, "_create_vertexai_token_provider", lambda: provider)

    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base="https://vertex.example/v1",
        )
    )

    assert llm.config.api_key_provider is provider
    # The unrelated OpenAI key must not survive anywhere as the Vertex
    # credential: without the fix, it is both kept and handed to the client.
    assert llm.api_key == lm.DUMMY_API_KEY
    assert llm.client.api_key != "sk-unrelated-openai-key"


def test_vertexai_explicit_api_key_overrides_inherited_openai_key(monkeypatch):
    """An explicitly configured key still wins over ADC and over the env."""
    _clear_vertexai_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated-openai-key")

    def unexpected_provider():
        raise AssertionError("ADC provider should not be created")

    monkeypatch.setattr(lm, "_create_vertexai_token_provider", unexpected_provider)
    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base="https://vertex.example/v1",
            api_key="vertex-token",
        )
    )

    assert llm.api_key == "vertex-token"
    assert llm.config.api_key_provider is None


def test_vertexai_ignores_lowercase_openai_api_key_env(monkeypatch):
    """Env keys are read case-insensitively, so provenance -- not value --
    decides whether the key is a caller-supplied Vertex credential."""
    _clear_vertexai_env(monkeypatch)
    monkeypatch.setenv("openai_api_key", "sk-lowercase-openai-key")
    provider = lambda: "adc-token"  # noqa: E731
    monkeypatch.setattr(lm, "_create_vertexai_token_provider", lambda: provider)

    config = lm.OpenAIGPTConfig(
        chat_model="vertexai/google/gemini-3-flash",
        api_base="https://vertex.example/v1",
    )
    # Guard against a vacuous test: the lowercase env var must really land in
    # the config, else there would be no inherited key to ignore.
    assert config.api_key == "sk-lowercase-openai-key"

    llm = lm.OpenAIGPT(config)

    assert llm.config.api_key_provider is provider
    assert llm.api_key == lm.DUMMY_API_KEY


def test_vertexai_explicit_key_matching_openai_env_is_honored(monkeypatch):
    """An explicit key is honored even when it equals the OpenAI env key."""
    _clear_vertexai_env(monkeypatch)
    shared = "same-token-in-both-places"
    monkeypatch.setenv("OPENAI_API_KEY", shared)

    def unexpected_provider():
        raise AssertionError("ADC provider should not be created")

    monkeypatch.setattr(lm, "_create_vertexai_token_provider", unexpected_provider)
    llm = lm.OpenAIGPT(
        lm.OpenAIGPTConfig(
            chat_model="vertexai/google/gemini-3-flash",
            api_base="https://vertex.example/v1",
            api_key=shared,
        )
    )

    assert llm.api_key == shared
    assert llm.config.api_key_provider is None


def test_vertexai_key_assigned_after_construction_is_honored(monkeypatch):
    """Assigning api_key post-construction counts as caller configuration."""
    _clear_vertexai_env(monkeypatch)

    def unexpected_provider():
        raise AssertionError("ADC provider should not be created")

    monkeypatch.setattr(lm, "_create_vertexai_token_provider", unexpected_provider)
    config = lm.OpenAIGPTConfig(
        chat_model="vertexai/google/gemini-3-flash",
        api_base="https://vertex.example/v1",
    )
    config.api_key = "late-token"
    llm = lm.OpenAIGPT(config)

    assert llm.api_key == "late-token"
    assert llm.config.api_key_provider is None


def test_vertexai_honors_custom_prefix_api_key(monkeypatch):
    """`<PREFIX>_API_KEY` from `create(prefix)` is provider-specific config."""
    _clear_vertexai_env(monkeypatch)
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-specific-token")

    def unexpected_provider():
        raise AssertionError("ADC provider should not be created")

    monkeypatch.setattr(lm, "_create_vertexai_token_provider", unexpected_provider)
    config = lm.OpenAIGPTConfig.create("vertex")(
        chat_model="vertexai/google/gemini-3-flash",
        api_base="https://vertex.example/v1",
    )
    # Guard against a vacuous test: the prefixed env var must really land in
    # the config, else there would be no provider-specific key to honor.
    assert config.api_key == "vertex-specific-token"

    llm = lm.OpenAIGPT(config)

    assert llm.api_key == "vertex-specific-token"
    assert llm.config.api_key_provider is None


def test_vertexai_custom_prefix_without_key_still_uses_adc(monkeypatch):
    """A custom prefix with no key set must still fall back to ADC."""
    _clear_vertexai_env(monkeypatch)
    provider = lambda: "adc-token"  # noqa: E731
    monkeypatch.setattr(lm, "_create_vertexai_token_provider", lambda: provider)

    config = lm.OpenAIGPTConfig.create("vertex")(
        chat_model="vertexai/google/gemini-3-flash",
        api_base="https://vertex.example/v1",
    )
    assert config.api_key == lm.DUMMY_API_KEY

    llm = lm.OpenAIGPT(config)

    assert llm.config.api_key_provider is provider
    assert llm.api_key == lm.DUMMY_API_KEY


def test_api_key_provenance_survives_model_copy(monkeypatch):
    """`model_copy` must preserve, and `update` must confer, key provenance."""
    _clear_vertexai_env(monkeypatch)

    supplied = lm.OpenAIGPTConfig(
        chat_model="vertexai/google/gemini-3-flash", api_key="explicit"
    )
    assert supplied.model_copy(update={"organization": "org"})._api_key_was_supplied

    inherited = lm.OpenAIGPTConfig(chat_model="vertexai/google/gemini-3-flash")
    assert not inherited._api_key_was_supplied
    assert inherited.model_copy(update={"api_key": "explicit"})._api_key_was_supplied


def test_vertexai_token_provider_is_memoized(monkeypatch):
    """ADC discovery runs once per process, so clients stay cacheable."""
    _clear_vertexai_env(monkeypatch)
    created = []

    class StubTokenProvider:
        def __init__(self) -> None:
            created.append(self)

        def __call__(self) -> str:
            return "stub-token"

    monkeypatch.setattr(lm, "_VertexAITokenProvider", StubTokenProvider)
    lm._create_vertexai_token_provider.cache_clear()
    try:
        configs = [
            lm.OpenAIGPTConfig(
                chat_model="vertexai/google/gemini-3-flash",
                api_base="https://vertex.example/v1",
            )
            for _ in range(2)
        ]
        providers = [lm.OpenAIGPT(config).config.api_key_provider for config in configs]

        assert len(created) == 1
        assert providers[0] is providers[1] is created[0]
    finally:
        lm._create_vertexai_token_provider.cache_clear()
