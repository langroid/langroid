"""
Isolation tests for the `vertexai/` route: proof that no `OPENAI_`-prefixed
value -- from the config, the transport, or the `openai` SDK's own
environment reads -- reaches Google's endpoint, and that everything the
caller DID configure survives the rebuild.

The poisoned-environment and wire-level tests here originate with
@BLVCK-MAMBA-6 (PR #1175). Split out of test_vertexai_routing.py to keep both
files under the repo's 1000-line limit; endpoint construction, credential
precedence and id validation stay there, and the ADC provider itself is in
test_vertexai_adc.py.
"""

import os
from typing import List

import httpx
import pytest

from langroid.language_models import openai_gpt
from langroid.language_models.openai_gpt import (
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


def test_openai_sdk_env_channels_do_not_reach_google(
    monkeypatch, project_env, fake_adc
):
    """
    The `openai` SDK reads its OWN `OPENAI_*` variables inside
    `OpenAI.__init__`, for arguments langroid does not pass. Those bypass the
    clean-config mechanism entirely -- they are never config fields, so no
    deny-list can see them. `OPENAI_PROJECT_ID` becomes an `OpenAI-Project`
    header and `OPENAI_CUSTOM_HEADERS` becomes arbitrary headers, both on
    every request. This is the #1165 hazard one layer below the config.
    """
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_leaked_openai_project")
    monkeypatch.setenv(
        "OPENAI_CUSTOM_HEADERS", "X-Leaked-Custom: sk-custom-header-secret"
    )
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
        OpenAIGPTConfig(
            chat_model=VERTEX_MODEL,
            http_client_factory=lambda: http_client,
            use_cached_client=False,
        )
    )
    with temporary_settings(Settings(cache=False)):
        llm.chat("hi")

    assert captured, "No HTTP request was captured"
    req = captured[-1]
    assert req.url.host.endswith("aiplatform.googleapis.com")
    joined = "\n".join(f"{k}: {v}" for k, v in req.headers.items())
    assert "proj_leaked_openai_project" not in joined
    assert "sk-custom-header-secret" not in joined
    assert req.headers.get("openai-project") is None
    assert req.headers.get("x-leaked-custom") is None

    # the variables are only hidden during client construction, not unset
    assert os.environ["OPENAI_PROJECT_ID"] == "proj_leaked_openai_project"
    assert "OPENAI_CUSTOM_HEADERS" in os.environ


def test_openai_sdk_env_still_applies_to_a_real_openai_route(monkeypatch):
    """The scrub is scoped to the vertexai/ route and must not leak out."""
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj_mine")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mine")
    llm = OpenAIGPT(OpenAIGPTConfig(chat_model="gpt-4o", use_cached_client=False))
    assert llm.client is not None
    assert llm.client.project == "proj_mine"


@pytest.mark.parametrize(
    "bad_model", ["vertexai//mistral-instruct-v0.2", "vertexai//hf", "vertexai//"]
)
def test_formatter_suffix_cannot_strip_the_route(project_env, fake_adc, bad_model):
    """
    `model//formatter` is split AFTER the route guards used to run, and
    `"vertexai//hf".split("//")[0]` is the bare model `vertexai` -- which
    routes to api.openai.com. On a VertexAIConfig that handed a
    VERTEXAI_API_KEY to OpenAI's client. The guards now run after the split.
    """
    with pytest.raises(ValueError, match="vertexai/"):
        OpenAIGPT(VertexAIConfig(chat_model=bad_model))


@pytest.mark.parametrize(
    "bad_model", ["vertexai//mistral-instruct-v0.2", "vertexai//hf"]
)
def test_formatter_suffix_on_a_plain_config_is_refused_too(
    monkeypatch, project_env, fake_adc, bad_model
):
    """
    Same input on a plain OpenAIGPTConfig leaks no Google credential -- there
    is none -- but it used to route silently to api.openai.com with the OpenAI
    key, which is never what someone typing `vertexai` meant.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    with pytest.raises(ValueError, match="left no model behind"):
        OpenAIGPT(OpenAIGPTConfig(chat_model=bad_model))


def test_formatter_suffix_on_a_real_vertex_model_still_works(project_env, fake_adc):
    """The `//` suffix is legitimate when the route survives the split."""
    llm = OpenAIGPT(VertexAIConfig(chat_model=f"{VERTEX_MODEL}//mistral-instruct-v0.2"))
    assert llm.is_vertexai is True
    assert llm.config.chat_model == "google/gemini-2.5-flash"
    assert llm.config.formatter == "mistral-instruct-v0.2"
    assert "aiplatform.googleapis.com" in (llm.api_base or "")


def test_formatter_suffix_override_cannot_strip_the_route(project_env, fake_adc):
    """Same hole via the global override, which is how it was reachable."""
    with temporary_settings(Settings(chat_model="vertexai//hf")):
        with pytest.raises(ValueError, match="left no model behind"):
            OpenAIGPT(VertexAIConfig(chat_model=VERTEX_MODEL))


def test_dropped_api_key_set_in_code_is_logged(project_env, fake_adc, caplog):
    """
    Dropping a token the caller passed in code, then silently falling back to
    ADC, is indistinguishable from a bug. Log the field name (never the value).
    """
    with caplog.at_level("WARNING"):
        llm = OpenAIGPT(OpenAIGPTConfig(chat_model=VERTEX_MODEL, api_key="sk-my-token"))
    assert "api_key" in caplog.text
    assert "sk-my-token" not in caplog.text
    assert llm.config.api_key == DUMMY_API_KEY
    assert llm.config.api_key_provider is not None


def test_every_openai_config_field_is_classified(project_env):
    """
    Guard: the conversion carries fields by default, so a new
    `OpenAIGPTConfig` field is carried onto the Vertex route automatically.
    If it is `OPENAI_`-env-populatable and influences the request, that is a
    leak. Every field must therefore be either dropped or listed here as
    reviewed-and-safe, so adding one forces the decision.
    """
    # Reviewed as safe to carry: none of these can change the HOST the request
    # goes to, attach a credential or identifier to it, or weaken its
    # transport. Two caveats, stated rather than glossed:
    #   - `params` is carried, but its free-form `extra_body` and its `user`
    #     identifier are cleared (see test_openai_params_extra_body_is_cleared
    #     and test_openai_params_user_is_cleared).
    #   - `formatter`, `use_completion_for_chat` and `use_chat_for_completion`
    #     do change which PATH on the Vertex host is called
    #     (/completions vs /chat/completions) and how the prompt is rendered.
    #     They are carried anyway: they are legitimate generation settings, and
    #     `OPENAI_FORMATTER` triggering a HuggingFace template lookup is
    #     pre-existing langroid behaviour on every route, not something this
    #     one introduces.
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
