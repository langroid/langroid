"""
Tests for `_adc_access_token`, the Google Application Default Credentials
provider installed on the `vertexai/` route.

Split out of test_vertexai_routing.py: these exercise the provider in
isolation, with a fake `google.auth` on sys.modules, and need none of that
module's routing fixtures.
"""

import sys
import threading
import time
import types
from typing import Any, Dict, List, Optional

import pytest

from langroid.language_models import openai_gpt


@pytest.fixture(autouse=True)
def reset_adc_state():
    """Each test starts with no cached credentials, and leaves none behind."""
    openai_gpt._adc_credentials = None
    yield
    openai_gpt._adc_credentials = None


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
    monkeypatch, counters: Dict[str, int], creds: _FakeCreds, delay: float = 0.0
) -> None:
    """Put a minimal fake `google.auth` on sys.modules for the test's duration.

    `delay` widens the window in which concurrent callers can race.
    """

    def _default(scopes: Any = None) -> Any:
        counters["default"] += 1
        if delay:
            time.sleep(delay)
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


def test_adc_resolves_credentials_once_under_concurrent_callers(monkeypatch):
    """
    The provider is called per request, so several threads can enter it at
    once. The lock must make credential resolution happen exactly once.

    The no-lock control proves this test is not vacuous: without mutual
    exclusion the same code resolves credentials once per thread.
    """

    class _NoLock:
        def __enter__(self) -> "_NoLock":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    def run(lock: Any) -> Dict[str, int]:
        counters = {"default": 0, "refresh": 0}
        creds = _FakeCreds(counters, valid=True)
        _install_fake_google_auth(monkeypatch, counters, creds, delay=0.02)
        monkeypatch.setattr(openai_gpt, "_adc_lock", lock)
        openai_gpt._adc_credentials = None
        tokens: List[str] = []
        errors: List[str] = []

        def worker() -> None:
            try:
                tokens.append(openai_gpt._adc_access_token())
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], errors
        assert len(tokens) == 12
        return counters

    with_lock = run(threading.Lock())
    assert with_lock["default"] == 1, with_lock

    without_lock = run(_NoLock())
    assert without_lock["default"] > 1, (
        "the no-lock control did not reproduce the race, so the locked case "
        f"proves nothing: {without_lock}"
    )


def test_adc_empty_token_raises(monkeypatch):
    """
    A credentials object that refreshes to no token must fail loudly rather
    than installing an empty bearer token on every request.
    """
    counters = {"default": 0, "refresh": 0}
    creds = _FakeCreds(counters, valid=True)
    creds.token = ""
    _install_fake_google_auth(monkeypatch, counters, creds)

    with pytest.raises(ValueError, match="empty access token"):
        openai_gpt._adc_access_token()


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
