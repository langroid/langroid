"""
Tests for the `vertexai/` route: a `vertexai/` model must build a *clean*
config, so that no `OPENAI_`-prefixed environment value can reach Google's
endpoint, while everything the caller configured is preserved.

The poisoned-environment and wire-level tests here originate with
@BLVCK-MAMBA-6 (PR #1175); the config-preservation, env-validation and
ADC-caching tests were added when that PR was taken over.
"""

import pytest

from langroid.language_models import openai_gpt
from langroid.language_models.openai_gpt import (
    DEFAULT_VERTEXAI_LOCATION,
    DUMMY_API_KEY,
    OpenAIGPT,
    OpenAIGPTConfig,
    VertexAIConfig,
)
from langroid.pydantic_v1 import ValidationError
from langroid.utils.configuration import (
    Settings,
    _global_settings,
    temporary_settings,
)

FAKE_ADC_TOKEN = "ya29.fake-adc-token-for-testing"
FAKE_OPENAI_KEY = "sk-malicious-openai-key"
FAKE_OPENAI_HEADERS_JSON = (
    '{"Authorization": "Bearer sk-malicious-header", "Custom-OpenAI": "1"}'
)
FAKE_OPENAI_ORG = "org-malicious-leak"
FAKE_OPENAI_BASE = "https://malicious.openai.endpoint/v1"

VERTEX_MODEL = "vertexai/google/gemini-2.5-flash"


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Clear ADC memoization, provider env vars, and settings.chat_model."""
    openai_gpt._adc_credentials = None
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_HEADERS",
        "OPENAI_ORGANIZATION",
        "OPENAI_API_BASE",
        "VERTEXAI_API_KEY",
        "VERTEXAI_PROJECT_ID",
        "VERTEXAI_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(var, raising=False)
    # The suite's conftest sets settings.chat_model globally; that override
    # takes precedence over the config inside OpenAIGPT.__init__ and would
    # hide the vertexai/ route under test.
    saved_chat_model = _global_settings.chat_model
    _global_settings.chat_model = ""
    yield
    _global_settings.chat_model = saved_chat_model
    openai_gpt._adc_credentials = None


@pytest.fixture
def poisoned_env(monkeypatch):
    """Simulate a process that has OpenAI env vars set (the common case)."""
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setenv("OPENAI_HEADERS", FAKE_OPENAI_HEADERS_JSON)
    monkeypatch.setenv("OPENAI_ORGANIZATION", FAKE_OPENAI_ORG)
    monkeypatch.setenv("OPENAI_API_BASE", FAKE_OPENAI_BASE)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    yield


@pytest.fixture
def fake_adc(monkeypatch):
    """Replace the ADC token provider with one returning a known token.

    Whether it was actually called is asserted on the wire, via the
    Authorization header, rather than by recording calls here.
    """

    def _provider() -> str:
        return FAKE_ADC_TOKEN

    monkeypatch.setattr(openai_gpt, "_adc_access_token", _provider)


@pytest.fixture
def project_env(monkeypatch):
    """A valid project, with no location set (so the default applies)."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    yield


# ---------------------------------------------------------------------------
# 3. Credential precedence
# ---------------------------------------------------------------------------


def test_adc_used_when_no_key_or_provider(project_env, fake_adc):
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert llm.config.api_key == DUMMY_API_KEY
    assert llm.config.api_key_provider is not None


def test_explicit_api_key_skips_adc(project_env, fake_adc):
    llm = OpenAIGPT(
        VertexAIConfig(chat_model=VERTEX_MODEL, api_key="my-explicit-token")
    )
    assert llm.config.api_key == "my-explicit-token"
    assert llm.config.api_key_provider is None


def test_vertexai_api_key_env_skips_adc(monkeypatch, project_env, fake_adc):
    monkeypatch.setenv("VERTEXAI_API_KEY", "token-from-env")
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert llm.config.api_key == "token-from-env"
    assert llm.config.api_key_provider is None


def test_custom_api_key_provider_takes_precedence(project_env, fake_adc):
    def custom_provider() -> str:
        return "custom-provider-token"

    llm = OpenAIGPT(
        VertexAIConfig(chat_model=VERTEX_MODEL, api_key_provider=custom_provider)
    )
    assert llm.config.api_key_provider is custom_provider


# ---------------------------------------------------------------------------
# 4. Endpoint construction and injection defence
# ---------------------------------------------------------------------------


def test_vertexai_endpoint_built_correctly(monkeypatch, fake_adc):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-abc")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert llm.api_base == (
        "https://asia-northeast1-aiplatform.googleapis.com/v1beta1/"
        "projects/proj-abc/locations/asia-northeast1/endpoints/openapi"
    )
    # the route prefix is stripped; the publisher/model id is what is sent
    assert llm.config.chat_model == "google/gemini-2.5-flash"


def test_default_location_when_unset(project_env, fake_adc):
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert f"https://{DEFAULT_VERTEXAI_LOCATION}-aiplatform" in (llm.api_base or "")


def test_explicit_api_base_is_honored(monkeypatch, project_env, fake_adc):
    """
    An explicit `api_base` (or `VERTEXAI_API_BASE`) targets a private/PSC
    endpoint. It can only have been set deliberately -- the conversion never
    carries `api_base` over from an OPENAI_-prefixed config -- so it must win
    over the constructed regional URL rather than being silently ignored.
    """
    private = "https://my-psc-endpoint.internal/v1beta1/projects/p/x/openapi"
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL, api_base=private))
    assert llm.api_base == private

    monkeypatch.setenv("VERTEXAI_API_BASE", private)
    llm2 = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert llm2.api_base == private


def test_explicit_api_base_is_not_inherited_from_openai_env(poisoned_env, fake_adc):
    """OPENAI_API_BASE must not become the Vertex endpoint via that door."""
    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert "malicious.openai.endpoint" not in (llm.api_base or "")
    assert "aiplatform.googleapis.com" in (llm.api_base or "")


def test_project_and_location_precedence(monkeypatch, fake_adc):
    """
    docs/notes/gemini.md promises: explicit config > VERTEXAI_* >
    GOOGLE_CLOUD_*. Pin all three tiers, not just the lowest.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-google-env")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-west1")
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "proj-vertexenv")
    monkeypatch.setenv("VERTEXAI_LOCATION", "asia-northeast1")

    # VERTEXAI_* beats GOOGLE_CLOUD_*
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert "/projects/proj-vertexenv/" in (llm.api_base or "")
    assert "https://asia-northeast1-aiplatform." in (llm.api_base or "")

    # explicit config beats both
    llm2 = OpenAIGPT(
        VertexAIConfig(
            chat_model=VERTEX_MODEL,
            project_id="proj-explicit",
            location="europe-west4",
        )
    )
    assert "/projects/proj-explicit/" in (llm2.api_base or "")
    assert "https://europe-west4-aiplatform." in (llm2.api_base or "")


def test_google_cloud_location_used_when_no_vertexai_location(monkeypatch, fake_adc):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-google-env")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-west1")
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert "https://us-west1-aiplatform." in (llm.api_base or "")
    assert "/locations/us-west1/" in (llm.api_base or "")


def test_completion_model_also_loses_the_prefix(poisoned_env, fake_adc):
    """
    On the settings.chat_model override path `completion_model` is set to the
    prefixed name; it must be stripped alongside `chat_model`, or the
    completions endpoint is asked for a model id that does not exist.
    """
    with temporary_settings(Settings(chat_model=VERTEX_MODEL)):
        llm = OpenAIGPT(OpenAIGPTConfig(chat_model="gpt-4o"))
        assert llm.config.chat_model == "google/gemini-2.5-flash"
        assert not llm.config.completion_model.startswith("vertexai/")


def test_deliberately_set_dropped_field_is_logged(project_env, fake_adc, caplog):
    """A dropped api_base/headers/organization must not vanish in silence."""
    with caplog.at_level("WARNING"):
        llm = OpenAIGPT(
            OpenAIGPTConfig(
                chat_model=VERTEX_MODEL,
                api_base="https://my-private-psc.example/v1",
            )
        )
    assert "api_base" in caplog.text
    # and it really was dropped, not honored
    assert "my-private-psc" not in (llm.api_base or "")
    assert "aiplatform.googleapis.com" in (llm.api_base or "")


def test_cached_client_path_isolates_projects(monkeypatch, fake_adc):
    """
    The default path is `use_cached_client=True`, which the wire tests do not
    exercise. Two projects must not share a client, or one project's requests
    would go to the other's endpoint.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-one")
    a = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-two")
    b = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))

    assert a.client is not None and b.client is not None
    assert "/projects/proj-one/" in str(a.client.base_url)
    assert "/projects/proj-two/" in str(b.client.base_url)
    assert a.client is not b.client

    # the same config reuses its cached client
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-one")
    c = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert c.client is a.client


def test_gcp_project_env_is_a_fallback(monkeypatch, fake_adc):
    monkeypatch.setenv("GCP_PROJECT", "proj-from-gcp-var")
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert "/projects/proj-from-gcp-var/" in (llm.api_base or "")


def test_missing_project_raises_with_guidance(fake_adc):
    with pytest.raises(ValueError, match="GCP project is required"):
        OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))


@pytest.mark.parametrize(
    "bad",
    [
        "attacker.com/",
        "user@host",
        "GLOBAL",
        "has spaces",
        "Has-Uppercase",
        "under_score",
        "",
        "a.b",
        "a%2Fb",
        "a:b",
    ],
)
def test_invalid_project_id_rejected(bad):
    with pytest.raises(ValidationError, match="Invalid Vertex AI project_id"):
        VertexAIConfig(chat_model=VERTEX_MODEL, project_id=bad)


@pytest.mark.parametrize(
    "bad",
    ["attacker.com/", "user@host", "GLOBAL", "global", "has spaces", "a.b"],
)
def test_invalid_location_rejected(bad):
    with pytest.raises(ValidationError, match="Vertex AI location"):
        VertexAIConfig(chat_model=VERTEX_MODEL, location=bad)


def test_valid_project_and_location_accepted():
    cfg = VertexAIConfig(
        chat_model=VERTEX_MODEL, project_id="my-project-1", location="europe-west4"
    )
    assert cfg.project_id == "my-project-1"
    assert cfg.location == "europe-west4"


@pytest.mark.parametrize(
    "bad", ["us-central1\n", "\nus-central1", "us-central1 ", "-", "--", "a-", "-a"]
)
def test_trailing_whitespace_and_bare_hyphens_rejected(bad):
    """
    `re.match(r"...$")` accepts a trailing newline, so "us-central1\\n" would
    pass straight into the URL; validation must use `fullmatch`.
    """
    with pytest.raises(ValidationError, match="Invalid Vertex AI location"):
        VertexAIConfig(chat_model=VERTEX_MODEL, location=bad)


@pytest.mark.parametrize(
    "bad_location",
    ["attacker.example.com/v1/x", "a.b", "x/../y", "global"],
)
def test_env_location_cannot_reach_the_endpoint_url(
    monkeypatch, project_env, fake_adc, bad_location
):
    """
    `GOOGLE_CLOUD_LOCATION` bypasses the pydantic field validators entirely,
    yet is interpolated into the URL authority. It must be validated at the
    point of use, or it could redirect the request -- and the ADC bearer token
    -- to an arbitrary host.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", bad_location)
    with pytest.raises(ValueError, match="Vertex AI location"):
        OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))


@pytest.mark.parametrize("env_var", ["GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"])
def test_env_project_cannot_reach_the_endpoint_url(monkeypatch, fake_adc, env_var):
    """Same for the project, which lands in the URL path."""
    monkeypatch.setenv(env_var, "p/../../../evil")
    with pytest.raises(ValueError, match="Vertex AI project_id"):
        OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))


def test_vertexai_route_is_not_treated_as_a_gemini_route(project_env, fake_adc):
    """
    `vertexai/google/gemini-*` must take the Vertex branch, not the Gemini
    one: `GEMINI_MODEL_PREFIXES` includes `google/gemini-`, which the model id
    matches once the route prefix is stripped.
    """
    llm = OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))
    assert llm.is_vertexai is True
    assert llm.is_gemini is False
    assert "generativelanguage.googleapis.com" not in (llm.api_base or "")
    assert "aiplatform.googleapis.com" in (llm.api_base or "")
