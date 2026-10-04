"""
Tests for Vertex AI routing: proof that OPENAI_* environment variables
never leak into Vertex AI requests.

These tests are the whole point of the PR — if they pass, the design
is validated end-to-end.
"""

from typing import List

import httpx
import pytest

from langroid.language_models import openai_gpt
from langroid.language_models.openai_gpt import (
    OpenAIGPT,
    OpenAIGPTConfig,
    VertexAIConfig,
)
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


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset ADC memoization, env vars, and settings.chat_model before each test."""
    openai_gpt._adc_provider_instance = None
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_HEADERS",
        "OPENAI_ORGANIZATION",
        "OPENAI_API_BASE",
        "VERTEXAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # The langroid test suite sets settings.chat_model globally in conftest;
    # that override takes precedence over the config inside OpenAIGPT.__init__
    # and hides the vertexai/ route we are trying to test. Save and clear it
    # for the duration of each test, then restore it.
    saved_chat_model = _global_settings.chat_model
    _global_settings.chat_model = ""
    yield
    _global_settings.chat_model = saved_chat_model
    openai_gpt._adc_provider_instance = None


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
    """Replace _get_adc_provider with a fake that returns a known token."""
    invoked: List[bool] = []

    def _provider() -> str:
        invoked.append(True)
        return FAKE_ADC_TOKEN

    monkeypatch.setattr(openai_gpt, "_get_adc_provider", lambda: _provider)
    yield invoked


# ---------------------------------------------------------------------------
# 1. The poisoned-environment tests (the critical ones)
# ---------------------------------------------------------------------------


def test_poisoned_env_does_not_leak_into_vertexai(poisoned_env, fake_adc):
    """An OpenAI-poisoned env must not populate a VertexAIConfig's fields."""
    cfg = OpenAIGPTConfig(chat_model="vertexai/google/gemini-2.5-flash")
    llm = OpenAIGPT(cfg)

    # Route must have been converted to a clean VertexAIConfig
    assert isinstance(llm.config, VertexAIConfig)

    # Endpoint is Google, not the OpenAI base from env
    assert "aiplatform.googleapis.com" in (llm.api_base or "")
    assert "malicious.openai.endpoint" not in (llm.api_base or "")

    # No OpenAI organization or headers inherited
    assert llm.config.organization == ""
    assert llm.config.headers == {}

    # The ADC provider should be installed (it's lazily invoked by the
    # SDK when a request is actually made, not at construction time)
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

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    cfg = VertexAIConfig(
        chat_model="vertexai/google/gemini-2.5-flash",
        http_client_factory=lambda: http_client,
        use_cached_client=False,
    )
    llm = OpenAIGPT(cfg)
    llm.chat("hi")

    assert captured, "No HTTP request was captured"
    req = captured[-1]

    # Host is Google
    assert req.url.host.endswith("aiplatform.googleapis.com")
    assert "malicious.openai.endpoint" not in str(req.url)

    # No leaked OpenAI strings in any header
    joined = "\n".join(f"{k}: {v}" for k, v in req.headers.items()).lower()
    assert FAKE_OPENAI_KEY.lower() not in joined
    assert "sk-malicious-header" not in joined
    assert "org-malicious-leak" not in joined
    assert "custom-openai" not in joined

    # The Authorization header must carry the ADC token
    auth = req.headers.get("authorization") or ""
    assert FAKE_ADC_TOKEN in auth, f"Authorization was: {auth!r}"


def test_settings_override_to_vertexai_does_not_leak(poisoned_env, fake_adc):
    """Switching to vertexai/ via settings.chat_model must not leak OpenAI key."""
    with temporary_settings(Settings(chat_model="vertexai/google/gemini-2.5-flash")):
        cfg = OpenAIGPTConfig(
            chat_model="gpt-4o",
            api_key="sk-explicit-openai-key",
        )
        llm = OpenAIGPT(cfg)

        assert isinstance(llm.config, VertexAIConfig)
        assert "aiplatform.googleapis.com" in (llm.api_base or "")
        # The old OpenAI key must not be the api_key on the resulting client
        assert llm.config.api_key in (None, "")


# ---------------------------------------------------------------------------
# 2. ADC precedence tests
# ---------------------------------------------------------------------------


def test_adc_used_when_no_key_or_provider(poisoned_env, fake_adc):
    cfg = VertexAIConfig(chat_model="vertexai/google/gemini-2.5-flash")
    llm = OpenAIGPT(cfg)
    assert llm.config.api_key is None
    assert llm.config.api_key_provider is not None


def test_explicit_api_key_skips_adc(poisoned_env, fake_adc):
    cfg = VertexAIConfig(
        chat_model="vertexai/google/gemini-2.5-flash",
        api_key="my-explicit-token",
    )
    llm = OpenAIGPT(cfg)
    assert llm.config.api_key == "my-explicit-token"
    assert llm.config.api_key_provider is None


def test_custom_api_key_provider_takes_precedence(poisoned_env, fake_adc):
    def custom_provider() -> str:
        return "custom-provider-token"

    cfg = VertexAIConfig(
        chat_model="vertexai/google/gemini-2.5-flash",
        api_key_provider=custom_provider,
    )
    llm = OpenAIGPT(cfg)
    assert llm.config.api_key_provider is custom_provider


# ---------------------------------------------------------------------------
# 3. Endpoint validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_project",
    [
        "attacker.com/",
        "user@host",
        "GLOBAL",
        "has spaces",
        "Has-Uppercase",
        "under_score",
    ],
)
def test_invalid_project_id_rejected(bad_project):
    with pytest.raises(Exception):
        VertexAIConfig(
            chat_model="vertexai/google/gemini-2.5-flash",
            project_id=bad_project,
        )


@pytest.mark.parametrize(
    "bad_location",
    ["attacker.com/", "user@host", "GLOBAL", "has spaces", "Has-Uppercase"],
)
def test_invalid_location_rejected(bad_location):
    with pytest.raises(Exception):
        VertexAIConfig(
            chat_model="vertexai/google/gemini-2.5-flash",
            location=bad_location,
        )


def test_valid_project_and_location_accepted():
    cfg = VertexAIConfig(
        chat_model="vertexai/google/gemini-2.5-flash",
        project_id="my-project-1",
        location="europe-west4",
    )
    assert cfg.project_id == "my-project-1"
    assert cfg.location == "europe-west4"


# ---------------------------------------------------------------------------
# 4. Regression tests
# ---------------------------------------------------------------------------


def test_vertexai_config_does_not_inherit_openai_env(poisoned_env):
    """VertexAIConfig fields must remain defaults under OPENAI_* env vars."""
    cfg = VertexAIConfig(chat_model="vertexai/google/gemini-2.5-flash")
    assert cfg.api_key is None
    assert cfg.organization == ""
    assert cfg.headers == {}


def test_vertexai_endpoint_built_correctly(monkeypatch, fake_adc):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-abc")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    cfg = VertexAIConfig(chat_model="vertexai/google/gemini-2.5-flash")
    llm = OpenAIGPT(cfg)
    assert llm.api_base == (
        "https://asia-northeast1-aiplatform.googleapis.com/v1beta1/"
        "projects/proj-abc/locations/asia-northeast1/endpoints/openapi"
    )


def test_vertexai_constructor_headers_kwarg_stays_empty_under_openai_env(
    poisoned_env,
):
    """
    Reproduce the #1165 trap directly on VertexAIConfig: OPENAI_HEADERS
    is set in the environment, and headers={} is passed explicitly to
    the constructor. The explicit empty dict must survive — pydantic-
    settings must not merge an OPENAI_-prefixed value into a field
    passed via the constructor on a class whose env_prefix is VERTEXAI_.
    """
    cfg = VertexAIConfig(
        chat_model="vertexai/google/gemini-2.5-flash",
        headers={},
    )
    assert cfg.headers == {}
    assert "Authorization" not in cfg.headers
    assert "Custom-OpenAI" not in cfg.headers
    assert "org-malicious-leak" not in str(cfg.headers)


def test_non_vertexai_route_is_not_converted_to_vertexai_config(poisoned_env):
    """
    The clean-config conversion is gated on settings.chat_model (or
    config.chat_model) starting with 'vertexai/'. Other routes — notably
    langdb/ and portkey/, which write into config.headers themselves —
    must remain OpenAIGPTConfig instances, so their headers are never
    touched by the vertexai/ conversion.
    """
    cfg = OpenAIGPTConfig(chat_model="openrouter/anthropic/claude-3-5-haiku")
    llm = OpenAIGPT(cfg)

    # Not replaced by the vertexai/ conversion
    assert not isinstance(llm.config, VertexAIConfig)
    # Route reaches the openrouter base URL, not the OpenAI one
    assert "openrouter.ai" in (llm.api_base or "")
