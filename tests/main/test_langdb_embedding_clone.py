"""Offline request-level coverage for embedding provider reconstruction."""

import json
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from openai import AzureOpenAI, OpenAI

from langroid.embedding_models import models
from langroid.language_models.provider_params import LangDBParams


@pytest.fixture
def embedding_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[tuple[str, dict[str, Any]]]]:
    """Use real SDK serialization with an entirely in-memory transport."""
    requests: list[tuple[str, dict[str, Any]]] = []
    clients: list[OpenAI] = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    def create_client(**kwargs: Any) -> OpenAI:
        client = OpenAI(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
            max_retries=0,
        )
        clients.append(client)
        return client

    def create_azure_client(**kwargs: Any) -> AzureOpenAI:
        client = AzureOpenAI(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
            max_retries=0,
        )
        clients.append(client)
        return client

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 102]
    tokenizer.decode.return_value = "hello"
    monkeypatch.setattr(models, "load_dotenv", lambda: None)
    monkeypatch.setattr(models, "OpenAI", create_client)
    monkeypatch.setattr(models, "AzureOpenAI", create_azure_client)

    def get_tokenizer(model_name: str) -> MagicMock:
        assert model_name == "text-embedding-3-small"
        return tokenizer

    monkeypatch.setattr(models.tiktoken, "encoding_for_model", get_tokenizer)
    try:
        yield requests
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("project_id", ["", "project"])
@pytest.mark.parametrize("route", ["langdb/", "langdb/openai/", "", "openai/"])
@pytest.mark.parametrize("lifecycle", ["original", "reuse", "clone", "nested_clone"])
def test_embedding_provider_survives_reconstruction(
    embedding_requests: list[tuple[str, dict[str, Any]]],
    route: str,
    lifecycle: str,
    project_id: str,
) -> None:
    """Cloning or reusing a configuration must preserve the wire contract."""
    config = models.OpenAIEmbeddingsConfig(
        model_name=route + "text-embedding-3-small",
        api_key="offline-key",
        api_base="https://openai.invalid/v1",
        organization="",
        langdb_params=LangDBParams(
            api_key="offline-langdb-key",
            base_url="https://langdb.invalid",
            project_id=project_id,
        ),
    )
    original = models.OpenAIEmbeddings(config)
    model = original
    if lifecycle == "reuse":
        model = models.OpenAIEmbeddings(config)
    elif lifecycle == "clone":
        model = original.clone()
        assert model is not original
    elif lifecycle == "nested_clone":
        first_clone = original.clone()
        model = first_clone.clone()
        assert model is not first_clone
        assert model is not original

    assert model.embedding_fn()(["hello"]) == [[0.25, 0.75]]
    url, body = embedding_requests[-1]
    is_langdb = route.startswith("langdb/")
    assert url == (
        "https://langdb.invalid"
        + (f"/{project_id}/v1" if project_id else "")
        + "/embeddings"
        if is_langdb
        else "https://openai.invalid/v1/embeddings"
    )
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == (["hello"] if is_langdb else [[101, 102]])


def test_langdb_embedding_leaves_caller_config_unchanged(
    embedding_requests: list[tuple[str, dict[str, Any]]],
) -> None:
    config = models.OpenAIEmbeddingsConfig(
        model_name="langdb/openai/text-embedding-3-small",
        langdb_params=LangDBParams(api_key="offline-key"),
    )
    before = config.model_dump()
    model = models.OpenAIEmbeddings(config)
    assert config.model_dump() == before
    assert model.config is not config
    assert model.config.langdb_params is not config.langdb_params


def test_azure_embedding_request_unchanged(
    embedding_requests: list[tuple[str, dict[str, Any]]],
) -> None:
    model = models.AzureOpenAIEmbeddings(
        models.AzureOpenAIEmbeddingsConfig(
            api_key="offline-key",
            api_base="https://azure.invalid",
            deployment_name="deployment",
            model_name="text-embedding-3-small",
            api_version="2024-06-01",
        )
    )
    assert model.embedding_fn()(["hello"]) == [[0.25, 0.75]]
    url, body = embedding_requests[-1]
    assert url == (
        "https://azure.invalid/openai/deployments/deployment/embeddings"
        "?api-version=2024-06-01"
    )
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == [[101, 102]]


@pytest.mark.parametrize("route", ["openai/", "langdb/openai/"])
def test_weaviate_vectorizer_uses_normalized_model(
    embedding_requests: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    from langroid.vector_store.weaviatedb import WeaviateDB, WeaviateDBConfig

    client = MagicMock()
    client.collections.exists.return_value = False
    configure = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "weaviate",
        SimpleNamespace(
            connect_to_local=lambda **_: client,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "weaviate.classes.init",
        SimpleNamespace(
            Auth=MagicMock(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "weaviate.classes.config",
        SimpleNamespace(
            Configure=configure,
            VectorDistances=SimpleNamespace(COSINE="cosine"),
        ),
    )
    config = WeaviateDBConfig(
        docker=True,
        embedding=models.OpenAIEmbeddingsConfig(
            model_name=route + "text-embedding-3-small",
            api_key="offline-key",
            langdb_params=LangDBParams(api_key="offline-langdb-key"),
        ),
    )
    store = WeaviateDB(config)
    store.create_collection("documents")
    configure.Vectorizer.text2vec_openai.assert_called_once_with(
        model="text-embedding-3-small"
    )
