"""
Tests for the `vertexai/` route: a `vertexai/` model must build a *clean*
config, so that no `OPENAI_`-prefixed environment value can reach Google's
endpoint, while everything the caller configured is preserved.

The poisoned-environment and wire-level tests here originate with
@BLVCK-MAMBA-6 (PR #1175); the config-preservation, env-validation and
ADC-caching tests were added when that PR was taken over.
"""

import sys
import types
from typing import Any, Dict, List, Optional

import httpx
import pytest

from langroid.language_models import openai_gpt
from langroid.language_models.openai_gpt import (
    DEFAULT_VERTEXAI_LOCATION,
    DUMMY_API_KEY,
    OpenAIGPT,
    OpenAIGPTConfig,
    VertexAIConfig,
)
from langroid.language_models.provider_params import LangDBParams
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
    """Replace the ADC token provider with one returning a known token."""
    invoked: List[bool] = []

    def _provider() -> str:
        invoked.append(True)
        return FAKE_ADC_TOKEN

    monkeypatch.setattr(openai_gpt, "_adc_access_token", _provider)
    yield invoked


@pytest.fixture
def project_env(monkeypatch):
    """A valid project, with no location set (so the default applies)."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    yield


# ---------------------------------------------------------------------------
# 1. The poisoned-environment tests (the point of the feature)
# ---------------------------------------------------------------------------


def test_poisoned_env_does_not_leak_into_vertexai(poisoned_env, fake_adc):
    """An OpenAI-poisoned env must not populate a VertexAIConfig's fields."""
    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))

    assert isinstance(llm.config, VertexAIConfig)
    assert "aiplatform.googleapis.com" in (llm.api_base or "")
    assert "malicious.openai.endpoint" not in (llm.api_base or "")
    assert llm.config.organization == ""
    assert llm.config.headers == {}
    assert llm.config.api_key == DUMMY_API_KEY
    assert llm.api_key == DUMMY_API_KEY
    # ADC installed as the credential, invoked lazily per request
    assert llm.config.api_key_provider is not None


def test_poisoned_env_never_appears_on_the_wire(poisoned_env, fake_adc):
    """Wire-level: no OPENAI_* string can be found anywhere on the request."""
    captured: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAIGPT(
        VertexAIConfig(
            chat_model=VERTEX_MODEL,
            http_client_factory=lambda: http_client,
            use_cached_client=False,
        )
    )
    # `cache=False`: with the default redis response cache on, a prior run of
    # this test answers this prompt from the cache and no request is made at
    # all, so the assertions below would vacuously inspect nothing.
    with temporary_settings(Settings(cache=False)):
        llm.chat("hi")

    assert captured, "No HTTP request was captured"
    req = captured[-1]

    assert req.url.host.endswith("aiplatform.googleapis.com")
    assert "malicious.openai.endpoint" not in str(req.url)

    joined = "\n".join(f"{k}: {v}" for k, v in req.headers.items()).lower()
    assert FAKE_OPENAI_KEY.lower() not in joined
    assert "sk-malicious-header" not in joined
    assert "org-malicious-leak" not in joined
    assert "custom-openai" not in joined

    auth = req.headers.get("authorization") or ""
    assert FAKE_ADC_TOKEN in auth, f"Authorization was: {auth!r}"


def test_settings_override_to_vertexai_does_not_leak(poisoned_env, fake_adc):
    """Switching to vertexai/ via settings.chat_model must not leak the key."""
    with temporary_settings(Settings(chat_model=VERTEX_MODEL)):
        llm = OpenAIGPT(
            OpenAIGPTConfig(chat_model="gpt-4o", api_key="sk-explicit-openai-key")
        )
        assert isinstance(llm.config, VertexAIConfig)
        assert "aiplatform.googleapis.com" in (llm.api_base or "")
        assert llm.config.api_key == DUMMY_API_KEY
        # the explicit OpenAI key must not survive onto the Vertex route
        assert llm.api_key != "sk-explicit-openai-key"


def test_vertexai_config_does_not_inherit_openai_env(poisoned_env):
    """VertexAIConfig fields must remain defaults under OPENAI_* env vars."""
    cfg = VertexAIConfig(chat_model=VERTEX_MODEL)
    assert cfg.api_key == DUMMY_API_KEY
    assert cfg.organization == ""
    assert cfg.headers == {}
    assert cfg.api_base is None


def test_vertexai_constructor_headers_kwarg_stays_empty_under_openai_env(
    poisoned_env,
):
    """
    The #1165 trap, on the new class: `OPENAI_HEADERS` is set and `headers={}`
    is passed explicitly. pydantic-settings merges an env dict into a dict
    passed to the constructor, so this only holds because `OPENAI_HEADERS` is
    not an env source for a class whose `env_prefix` is `VERTEXAI_`.
    """
    cfg = VertexAIConfig(chat_model=VERTEX_MODEL, headers={})
    assert cfg.headers == {}


def test_vertexai_headers_env_is_honored(monkeypatch, project_env, fake_adc):
    """The VERTEXAI_-prefixed equivalent still works, by design."""
    monkeypatch.setenv("VERTEXAI_HEADERS", '{"X-Mine": "1"}')
    cfg = VertexAIConfig(chat_model=VERTEX_MODEL)
    assert cfg.headers.get("X-Mine") == "1"


def test_non_vertexai_route_is_not_converted_to_vertexai_config(poisoned_env):
    """
    The conversion is gated on the `vertexai/` prefix. Other routes -- notably
    langdb/ and portkey/, which write into `config.headers` themselves -- must
    stay `OpenAIGPTConfig` instances so their headers are never stripped.
    """
    llm = OpenAIGPT(OpenAIGPTConfig(chat_model="openrouter/anthropic/claude-3-5-haiku"))
    assert not isinstance(llm.config, VertexAIConfig)
    assert "openrouter.ai" in (llm.api_base or "")


def test_langdb_headers_survive():
    """A langdb/ route must keep the headers its own branch sets."""
    llm = OpenAIGPT(
        OpenAIGPTConfig(
            chat_model="langdb/openai/gpt-4o",
            langdb_params=LangDBParams(project_id="proj-langdb"),
        )
    )
    assert not isinstance(llm.config, VertexAIConfig)
    assert llm.config.headers.get("x-project-id") == "proj-langdb"


# ---------------------------------------------------------------------------
# 2. Caller configuration must survive the clean-config conversion
# ---------------------------------------------------------------------------


def test_user_config_fields_survive_vertexai_conversion(poisoned_env, fake_adc):
    """
    Rebuilding the config must not discard what the caller configured. Only
    the credential channels that `OPENAI_*` can populate are dropped.
    """
    cfg = OpenAIGPTConfig(
        chat_model=VERTEX_MODEL,
        temperature=0.123,
        max_output_tokens=4321,
        chat_context_length=77777,
        min_output_tokens=7,
        seed=99,
        use_cached_client=False,
        timeout=123,
    )
    llm = OpenAIGPT(cfg)

    assert isinstance(llm.config, VertexAIConfig)
    assert llm.config.temperature == 0.123
    assert llm.config.max_output_tokens == 4321
    assert llm.config.chat_context_length == 77777
    assert llm.config.min_output_tokens == 7
    assert llm.config.seed == 99
    assert llm.config.use_cached_client is False
    assert llm.config.timeout == 123


def test_user_config_fields_survive_settings_override(poisoned_env, fake_adc):
    """Same guarantee on the settings.chat_model override path."""
    with temporary_settings(Settings(chat_model=VERTEX_MODEL)):
        llm = OpenAIGPT(
            OpenAIGPTConfig(
                chat_model="gpt-4o",
                temperature=0.77,
                max_output_tokens=1234,
                use_cached_client=False,
            )
        )
        assert isinstance(llm.config, VertexAIConfig)
        assert llm.config.temperature == 0.77
        assert llm.config.max_output_tokens == 1234
        assert llm.config.use_cached_client is False


def test_explicit_api_key_provider_survives_conversion(poisoned_env):
    """
    `api_key_provider` is a callable, so it can only have been set in code --
    never by the environment. It must therefore survive the conversion, and
    must suppress the ADC fallback.
    """

    def custom_provider() -> str:
        return "custom-provider-token"

    llm = OpenAIGPT(
        OpenAIGPTConfig(chat_model=VERTEX_MODEL, api_key_provider=custom_provider)
    )
    assert isinstance(llm.config, VertexAIConfig)
    assert llm.config.api_key_provider is custom_provider


def test_openai_gpt_config_api_key_remains_a_required_str():
    """
    `OpenAIGPTConfig.api_key` is public API. The vertexai/ route must not widen
    it to `str | None`, which would push an Optional onto every downstream
    caller annotated `str`.
    """
    assert OpenAIGPTConfig.model_fields["api_key"].annotation is str
    with pytest.raises(ValidationError, match="api_key"):
        OpenAIGPTConfig(api_key=None)


def test_openai_gpt_init_docstring_is_intact():
    """Guards against inserting route logic above the __init__ docstring."""
    doc = OpenAIGPT.__init__.__doc__
    assert doc is not None and "config" in doc


def test_subclass_only_fields_are_reported_not_dropped_silently(
    project_env, fake_adc, caplog
):
    """
    `VertexAIConfig` has no fields of a custom `OpenAIGPTConfig` subclass, so
    they cannot be carried over -- but the caller must be told.
    """

    class MyConfig(OpenAIGPTConfig):
        my_custom_field: str = "keep-me"

    with caplog.at_level("WARNING"):
        llm = OpenAIGPT(MyConfig(chat_model=VERTEX_MODEL, temperature=0.5))

    assert isinstance(llm.config, VertexAIConfig)
    # the supported fields still come across
    assert llm.config.temperature == 0.5
    assert "my_custom_field" in caplog.text
    assert "MyConfig" in caplog.text


def test_vertexai_config_is_exported_from_language_models():
    """The docs tell callers to construct it directly, so it must be public."""
    import langroid.language_models as lm

    assert lm.VertexAIConfig is VertexAIConfig


def test_settings_override_away_from_vertexai_keeps_the_openai_key(monkeypatch):
    """
    Overriding `settings.chat_model` to a non-Vertex model must not leave the
    config stripped of its OpenAI credential: the conversion is keyed on the
    model actually in effect, so it must not fire here at all.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    with temporary_settings(Settings(chat_model="gpt-4o")):
        llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
        assert not isinstance(llm.config, VertexAIConfig)
        assert llm.api_key == "sk-real-openai-key"
        assert llm.config.api_key_provider is None


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


# ---------------------------------------------------------------------------
# 5. The ADC provider itself
# ---------------------------------------------------------------------------


class _FakeCreds:
    """Stand-in for google.auth credentials, tracking refresh calls."""

    def __init__(self, counters: Dict[str, int], valid: bool = True) -> None:
        self._counters = counters
        self.valid = valid
        self.token: Optional[str] = "tok-initial" if valid else None

    def refresh(self, request: Any) -> None:
        self._counters["refresh"] += 1
        self.valid = True
        self.token = f"tok-{self._counters['refresh']}"


def _install_fake_google_auth(
    monkeypatch, counters: Dict[str, int], creds: _FakeCreds
) -> None:
    """Put a minimal fake `google.auth` on sys.modules for the test's duration."""

    def _default(scopes: Any = None) -> Any:
        counters["default"] += 1
        return creds, "fake-project"

    requests_mod = types.ModuleType("google.auth.transport.requests")
    requests_mod.Request = lambda: object()  # type: ignore[attr-defined]
    transport_mod = types.ModuleType("google.auth.transport")
    transport_mod.requests = requests_mod  # type: ignore[attr-defined]
    exceptions_mod = types.ModuleType("google.auth.exceptions")

    class DefaultCredentialsError(Exception):
        pass

    exceptions_mod.DefaultCredentialsError = (  # type: ignore[attr-defined]
        DefaultCredentialsError
    )
    auth_mod = types.ModuleType("google.auth")
    auth_mod.default = _default  # type: ignore[attr-defined]
    auth_mod.transport = transport_mod  # type: ignore[attr-defined]
    auth_mod.exceptions = exceptions_mod  # type: ignore[attr-defined]
    google_mod = types.ModuleType("google")
    google_mod.auth = auth_mod  # type: ignore[attr-defined]

    for name, mod in [
        ("google", google_mod),
        ("google.auth", auth_mod),
        ("google.auth.transport", transport_mod),
        ("google.auth.transport.requests", requests_mod),
        ("google.auth.exceptions", exceptions_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_adc_credentials_are_cached_across_calls(monkeypatch):
    """
    The provider is called on every request, so it must not re-resolve
    credentials or refresh a still-valid token each time: google-auth already
    tracks expiry on the credentials object.
    """
    counters = {"default": 0, "refresh": 0}
    creds = _FakeCreds(counters, valid=True)
    _install_fake_google_auth(monkeypatch, counters, creds)
    openai_gpt._adc_credentials = None

    tokens = [openai_gpt._adc_access_token() for _ in range(5)]

    assert tokens == ["tok-initial"] * 5
    assert counters["default"] == 1, "credentials re-resolved per call"
    assert counters["refresh"] == 0, "a valid token was refreshed anyway"


def test_adc_refreshes_an_expired_token(monkeypatch):
    """When the cached token has expired, exactly one refresh happens."""
    counters = {"default": 0, "refresh": 0}
    creds = _FakeCreds(counters, valid=False)
    _install_fake_google_auth(monkeypatch, counters, creds)
    openai_gpt._adc_credentials = None

    first = openai_gpt._adc_access_token()
    assert first == "tok-1"
    assert counters["refresh"] == 1

    # token is valid now, so a second call must not refresh again
    assert openai_gpt._adc_access_token() == "tok-1"
    assert counters["refresh"] == 1
    assert counters["default"] == 1


def test_adc_missing_credentials_raises_with_guidance(monkeypatch):
    counters = {"default": 0, "refresh": 0}
    creds = _FakeCreds(counters)
    _install_fake_google_auth(monkeypatch, counters, creds)
    openai_gpt._adc_credentials = None

    exc = sys.modules["google.auth.exceptions"].DefaultCredentialsError

    def _boom(scopes: Any = None) -> Any:
        raise exc("no ADC")

    monkeypatch.setattr(sys.modules["google.auth"], "default", _boom)
    with pytest.raises(ValueError, match="application-default login"):
        openai_gpt._adc_access_token()
