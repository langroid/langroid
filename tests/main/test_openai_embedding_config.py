from unittest.mock import MagicMock

import pytest

from langroid.embedding_models import models


@pytest.mark.parametrize("env_organization", [None, "org-environment"])
@pytest.mark.parametrize("organization", ["org-explicit", ""])
def test_openai_embedding_organization_precedence(
    monkeypatch: pytest.MonkeyPatch,
    env_organization: str | None,
    organization: str,
) -> None:
    """An explicit organization wins; an empty value falls back to the env."""
    monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)
    # Create configuration before loading the environment, as with .env loading.
    config = models.OpenAIEmbeddingsConfig(
        api_key="test-key", organization=organization
    )
    if env_organization is not None:
        monkeypatch.setenv("OPENAI_ORGANIZATION", env_organization)
    monkeypatch.setattr(models, "load_dotenv", lambda: None)
    client = MagicMock()
    monkeypatch.setattr(models, "OpenAI", client)
    monkeypatch.setattr(models.tiktoken, "encoding_for_model", MagicMock())

    embedding = models.OpenAIEmbeddings(config)
    expected = organization or env_organization or ""
    assert client.call_args.kwargs["organization"] == expected
    assert embedding.config.organization == expected

    cloned = embedding.clone()
    assert cloned is not embedding
    assert cloned.config is not config
    assert cloned.config.organization == expected
    assert client.call_args.kwargs["organization"] == expected
