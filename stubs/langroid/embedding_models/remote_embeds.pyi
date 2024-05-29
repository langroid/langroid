from typing import Callable

import grpc
from _typeshed import Incomplete

import langroid.embedding_models.models as em
import langroid.embedding_models.protoc.embeddings_pb2 as embeddings_pb
import langroid.embedding_models.protoc.embeddings_pb2_grpc as embeddings_grpc
from langroid.mytypes import Embeddings as Embeddings

class RemoteEmbeddingRPCs(embeddings_grpc.EmbeddingServicer):
    embedding_fn: Incomplete
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        data_parallel: bool,
        device: str | None,
        devices: list[str] | None,
    ) -> None: ...
    def Embed(
        self, request: embeddings_pb.EmbeddingRequest, _: grpc.RpcContext
    ) -> embeddings_pb.BatchEmbeds: ...

class RemoteEmbeddingsConfig(em.SentenceTransformerEmbeddingsConfig):
    api_base: str
    port: int
    poll_delay: float
    max_retries: int

class RemoteEmbeddings(em.SentenceTransformerEmbeddings):
    config: Incomplete
    have_started_server: bool
    def __init__(self, config: RemoteEmbeddingsConfig = ...) -> None: ...
    def embedding_fn(self) -> Callable[[list[str]], Embeddings]: ...

async def serve(
    bind_address_base: str = "localhost",
    port: int = 50052,
    batch_size: int = 512,
    data_parallel: bool = False,
    device: str | None = None,
    devices: list[str] | None = None,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> None: ...
