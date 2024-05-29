from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import base
    from . import models
    from . import remote_embeds

    from .base import (
        EmbeddingModel,
        EmbeddingModelsConfig,
    )
    from .models import (
        OpenAIEmbeddings,
        OpenAIEmbeddingsConfig,
        SentenceTransformerEmbeddingsConfig,
        SentenceTransformerEmbeddings,
        embedding_model,
    )
    from .remote_embeds import (
        RemoteEmbeddingsConfig,
        RemoteEmbeddings,
    )

else:
    base = LazyLoad("langroid.embedding_models.base")
    models = LazyLoad("langroid.embedding_models.models")
    remote_embeds = LazyLoad("langroid.embedding_models.remote_embeds")

    RemoteEmbeddings = LazyLoad(
        "langroid.embedding_models.remote_embeds.RemoteEmbeddings"
    )
    RemoteEmbeddingsConfig = LazyLoad(
        "langroid.embedding_models.remote_embeds.RemoteEmbeddingsConfig"
    )
    EmbeddingModel = LazyLoad("langroid.embedding_models.base.EmbeddingModel")
    EmbeddingModelsConfig = LazyLoad(
        "langroid.embedding_models.base.EmbeddingModelsConfig"
    )
    OpenAIEmbeddings = LazyLoad("langroid.embedding_models.models.OpenAIEmbeddings")
    OpenAIEmbeddingsConfig = LazyLoad(
        "langroid.embedding_models.models.OpenAIEmbeddingsConfig"
    )
    SentenceTransformerEmbeddingsConfig = LazyLoad(
        "langroid.embedding_models.models.SentenceTransformerEmbeddingsConfig"
    )
    SentenceTransformerEmbeddings = LazyLoad(
        "langroid.embedding_models.models.SentenceTransformerEmbeddings"
    )
    embedding_model = LazyLoad("langroid.embedding_models.models.embedding_model")

__all__ = [
    "base",
    "models",
    "remote_embeds",
    "EmbeddingModel",
    "EmbeddingModelsConfig",
    "OpenAIEmbeddings",
    "OpenAIEmbeddingsConfig",
    "SentenceTransformerEmbeddingsConfig",
    "SentenceTransformerEmbeddings",
    "embedding_model",
    "RemoteEmbeddingsConfig",
    "RemoteEmbeddings",
]
