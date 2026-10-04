"""
Tests for the `vertexai/` route: a `vertexai/` model must build a *clean*
config, so that no `OPENAI_`-prefixed environment value can reach Google's
endpoint, while everything the caller configured is preserved.

The poisoned-environment and wire-level tests here originate with
@BLVCK-MAMBA-6 (PR #1175); the config-preservation, env-validation and
ADC-caching tests were added when that PR was taken over.
"""

from typing import List

import httpx
import pytest

from langroid.language_models import openai_gpt
from langroid.language_models.openai_gpt import (
    DEFAULT_VERTEXAI_LOCATION,
    DUMMY_API_KEY,
    OpenAICallParams,
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
    # Deliberately a plain OpenAIGPTConfig, not a VertexAIConfig: that routes
    # through _as_vertexai_config, so this test covers the drop-list -- the
    # feature's central mechanism. Built from a VertexAIConfig it would stay
    # green even if api_key/headers/organization/api_base were all carried.
    llm = OpenAIGPT(
        OpenAIGPTConfig(
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
    body = req.content.decode(errors="replace").lower()
    for leaked in (
        FAKE_OPENAI_KEY.lower(),
        "sk-malicious-header",
        "org-malicious-leak",
        "custom-openai",
        "malicious.openai.endpoint",
    ):
        assert leaked not in joined, f"{leaked!r} leaked in headers"
        assert leaked not in body, f"{leaked!r} leaked in the body"

    auth = req.headers.get("authorization") or ""
    assert FAKE_ADC_TOKEN in auth, f"Authorization was: {auth!r}"


@pytest.mark.asyncio
async def test_poisoned_env_never_appears_on_the_wire_async(poisoned_env, fake_adc):
    """
    The async client is built separately from the sync one and resolves the
    `api_key_provider` itself, so pin the same guarantee on `achat`.
    """
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

    # the factory may return (sync, async); achat uses the async one
    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # plain OpenAIGPTConfig, so this goes through the conversion too
    llm = OpenAIGPT(
        OpenAIGPTConfig(
            chat_model=VERTEX_MODEL,
            http_client_factory=lambda: (sync_client, async_client),
            use_cached_client=False,
        )
    )
    with temporary_settings(Settings(cache=False)):
        await llm.achat("hi")

    assert captured, "No HTTP request was captured on the async path"
    req = captured[-1]
    assert req.url.host.endswith("aiplatform.googleapis.com")
    joined = "\n".join(f"{k}: {v}" for k, v in req.headers.items()).lower()
    assert FAKE_OPENAI_KEY.lower() not in joined
    assert "sk-malicious-header" not in joined
    assert "org-malicious-leak" not in joined
    assert "custom-openai" not in joined
    assert FAKE_ADC_TOKEN in (req.headers.get("authorization") or "")


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
    """The VERTEXAI_-prefixed equivalent still works, and reaches the client."""
    monkeypatch.setenv("VERTEXAI_HEADERS", '{"X-Mine": "1"}')
    cfg = VertexAIConfig(chat_model=VERTEX_MODEL)
    assert cfg.headers.get("X-Mine") == "1"
    llm = OpenAIGPT(cfg)
    assert llm.config.headers.get("X-Mine") == "1"
    assert llm.client is not None
    assert llm.client.default_headers.get("X-Mine") == "1"


def test_openai_http_client_config_does_not_reach_the_vertex_client(
    monkeypatch, project_env, fake_adc
):
    """
    `OPENAI_HTTP_CLIENT_CONFIG` is spread into `httpx.Client(**config)`, so
    carrying it over would attach OpenAI-intended headers to the Vertex
    request, or proxy it (bearer token and all) through another host. This is
    #1165 through the transport rather than the config.

    Note this cannot be tested through `http_client_factory`: that takes an
    earlier branch and structurally bypasses `http_client_config`.
    """
    monkeypatch.setenv(
        "OPENAI_HTTP_CLIENT_CONFIG",
        '{"headers": {"X-OpenAI-Only": "sk-leak-marker"},'
        ' "proxy": "http://attacker.example:8080"}',
    )
    source = OpenAIGPTConfig(chat_model=VERTEX_MODEL)
    # the env var really does populate the OpenAI-prefixed config
    assert source.http_client_config is not None
    assert "sk-leak-marker" in str(source.http_client_config)

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL, use_cached_client=False))
    assert llm.config.http_client_config is None
    client = llm.client
    assert client is not None
    assert "sk-leak-marker" not in str(dict(client._client.headers))
    assert client._client._mounts == {}, "request would be proxied"


def test_openai_http_verify_ssl_does_not_disable_tls_to_google(
    monkeypatch, project_env, fake_adc
):
    """`OPENAI_HTTP_VERIFY_SSL=false` must not travel to the Google connection."""
    monkeypatch.setenv("OPENAI_HTTP_VERIFY_SSL", "false")
    assert OpenAIGPTConfig(chat_model=VERTEX_MODEL).http_verify_ssl is False

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL, use_cached_client=False))
    assert llm.config.http_verify_ssl is True
    client = llm.client
    assert client is not None
    transport = client._client._transport
    ssl_ctx = transport._pool._ssl_context
    assert ssl_ctx.check_hostname is True
    assert ssl_ctx.verify_mode != 0


def test_openai_chat_model_orig_cannot_hijack_the_route(
    monkeypatch, project_env, fake_adc
):
    """
    The provider branches key off `chat_model_orig`, and
    `OPENAI_CHAT_MODEL_ORIG` populates it. If it were carried over, a
    `vertexai/` model could be claimed by the Gemini branch instead: public
    endpoint, no ADC. The route must win regardless.
    """
    monkeypatch.setenv("OPENAI_CHAT_MODEL_ORIG", "google/gemini-2.5-flash")
    assert (
        OpenAIGPTConfig(chat_model=VERTEX_MODEL).chat_model_orig
        == "google/gemini-2.5-flash"
    )

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert llm.is_vertexai is True
    assert llm.is_gemini is False
    assert llm.chat_model_orig == VERTEX_MODEL
    assert "aiplatform.googleapis.com" in (llm.api_base or "")
    assert "generativelanguage.googleapis.com" not in (llm.api_base or "")
    assert llm.config.api_key_provider is not None


def test_openai_litellm_flag_cannot_hijack_the_route(
    monkeypatch, project_env, fake_adc
):
    """
    `litellm` is consulted as `startswith("litellm/") or config.litellm`, so
    `OPENAI_LITELLM=true` would divert a `vertexai/` model to the litellm
    adapter -- which also sidesteps the guard forbidding `api_key_provider`
    there, so ADC would be set and then silently unused.
    """
    monkeypatch.setenv("OPENAI_LITELLM", "true")
    assert OpenAIGPTConfig(chat_model=VERTEX_MODEL).litellm is True

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert llm.config.litellm is False
    assert llm.is_vertexai is True
    assert "aiplatform.googleapis.com" in (llm.api_base or "")
    assert llm.config.api_key_provider is not None


def test_openai_params_extra_body_is_cleared(monkeypatch, project_env, fake_adc):
    """
    `OPENAI_PARAMS` can set `params.extra_body` to an arbitrary dict, which is
    sent in the request body to Google -- the same hazard as OPENAI_HEADERS.
    The rest of `params` is generation behavior and must survive.
    """
    monkeypatch.setenv(
        "OPENAI_PARAMS",
        '{"extra_body": {"x_secret": "sk-from-openai-params"}, "top_p": 0.5}',
    )
    source = OpenAIGPTConfig(chat_model=VERTEX_MODEL)
    assert source.params is not None
    assert source.params.extra_body == {"x_secret": "sk-from-openai-params"}

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert llm.config.params is not None
    assert llm.config.params.extra_body is None
    # the legitimate generation settings are kept
    assert llm.config.params.top_p == 0.5


def test_extra_body_survives_on_a_direct_vertexai_config(project_env, fake_adc):
    """An extra_body set on a VertexAIConfig is meant for Vertex; keep it."""
    llm = OpenAIGPT(
        VertexAIConfig(
            chat_model=VERTEX_MODEL,
            params=OpenAICallParams(extra_body={"mine": 1}),
        )
    )
    assert llm.config.params is not None
    assert llm.config.params.extra_body == {"mine": 1}


def test_vertexai_config_refuses_a_non_vertex_model(monkeypatch, project_env):
    """
    The leak running backwards: a VertexAIConfig carries a credential meant
    for Google, and without the `vertexai/` prefix it would fall through to
    the generic branch -- `api_base=None`, i.e. api.openai.com -- and send it
    there. Reachable via the settings.chat_model override alone.
    """
    monkeypatch.setenv("VERTEXAI_API_KEY", "vertex-secret-token")

    with temporary_settings(Settings(chat_model="gpt-4o")):
        with pytest.raises(ValueError, match="not a vertexai/ route"):
            OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))

    # and directly, with no override involved
    with pytest.raises(ValueError, match="not a vertexai/ route"):
        OpenAIGPT(VertexAIConfig(chat_model="gpt-4o"))


def test_openai_params_user_is_cleared(monkeypatch, project_env, fake_adc):
    """
    `params.user` is an end-user identifier the provider logs, settable from
    OPENAI_PARAMS, and it travels in the request body -- the same class of
    value as `organization`, which is dropped outright.
    """
    monkeypatch.setenv(
        "OPENAI_PARAMS", '{"user": "victim-id", "logit_bias": {"1": 1.0}}'
    )
    assert OpenAIGPTConfig(chat_model=VERTEX_MODEL).params.user == "victim-id"

    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert llm.config.params is not None
    assert llm.config.params.user is None
    # generation behavior in the same env var is kept
    assert llm.config.params.logit_bias == {1: 1.0}


@pytest.mark.parametrize(
    "field,value",
    [
        ("http_client_config", {"proxy": "http://corp-proxy:3128"}),
        ("http_verify_ssl", False),
        ("litellm", True),
    ],
)
def test_dropped_transport_fields_are_logged(
    project_env, fake_adc, caplog, field, value
):
    """
    These are dropped for good reason, but a corporate-proxy user who set one
    in code would otherwise get an unreachable endpoint with no diagnostic.
    """
    with caplog.at_level("WARNING"):
        OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL, **{field: value}))
    assert field in caplog.text


def test_every_openai_config_field_is_classified(project_env):
    """
    Guard: the conversion carries fields by default, so a new
    `OpenAIGPTConfig` field is carried onto the Vertex route automatically.
    If it is `OPENAI_`-env-populatable and influences the request, that is a
    leak. Every field must therefore be either dropped or listed here as
    reviewed-and-safe, so adding one forces the decision.
    """
    # Reviewed as safe to carry: none of these can redirect the request,
    # attach anything to it, or weaken its transport. `params` is the one
    # partial case -- it is carried, but its free-form `extra_body` sub-field
    # is cleared (see test_openai_params_extra_body_is_cleared).
    carry_safe = {
        # model / generation behavior
        "chat_model",
        "completion_model",
        "temperature",
        "max_output_tokens",
        "min_output_tokens",
        "chat_context_length",
        "completion_context_length",
        "seed",
        "params",
        "stream",
        "async_stream_quiet",
        "formatter",
        "hf_formatter",
        "use_chat_for_completion",
        "use_completion_for_chat",
        "parallel_tool_calls",
        "supports_json_schema",
        "supports_strict_tools",
        "timeout",
        "retry_params",
        "cache_config",
        # callables: settable only in code, never from the environment
        "api_key_provider",
        "http_client_factory",
        "run_on_first_use",
        "streamer",
        "streamer_async",
        # client reuse: no effect on where the request goes
        "use_cached_client",
        # other providers' settings, inert on a vertexai/ route
        "litellm_proxy",
        "ollama",
        "langdb_params",
        "portkey_params",
        "thought_delimiters",
    }
    declared = set(OpenAIGPTConfig.model_fields)
    unclassified = declared - carry_safe - set(openai_gpt._VERTEXAI_DROPPED_FIELDS)
    assert not unclassified, (
        f"New OpenAIGPTConfig field(s) {sorted(unclassified)} are carried onto "
        "the vertexai/ route by default. Add each to "
        "_VERTEXAI_DROPPED_FIELDS if an OPENAI_* env var could set it and it "
        "affects the request, or to carry_safe in this test if it is inert."
    )
    # and nothing is classified that does not exist
    assert set(openai_gpt._VERTEXAI_DROPPED_FIELDS) <= declared
    assert carry_safe <= declared


def test_vertexai_config_type_is_set(project_env, fake_adc):
    llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL))
    assert llm.config.type == "vertexai"


@pytest.mark.parametrize("bad_model", ["vertexai/", "vertexai/foo"])
def test_malformed_vertexai_model_rejected(project_env, fake_adc, bad_model):
    """
    The route is documented as vertexai/<publisher>/<model>; enforce it, so a
    missing publisher cannot produce a live endpoint with an empty model id.

    `vertexai//` is deliberately not covered: `//` is langroid's existing
    `model//formatter` syntax and is consumed before this route is reached.
    """
    with pytest.raises(ValueError, match="expected vertexai/<publisher>/<model>"):
        OpenAIGPT(VertexAIConfig(chat_model=bad_model))


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
