"""
If run as a script, starts an RPC server which handles remote
embedding requests:

python3 -m langroid.embedding_models.remote_embeds `port`

where `port` is the port at which the service is exposed.  Currently,
supports insecure connections only, and this should NOT be exposed to
the internet.
"""

import subprocess
from concurrent import futures
from typing import Callable

import grpc
from fire import Fire

import langroid as lr
import langroid.embedding_models as em
import langroid.embedding_models.protoc.embeddings_pb2 as embeddings_pb
import langroid.embedding_models.protoc.embeddings_pb2_grpc as embeddings_grpc


class RemoteEmbeddingRPCs(embeddings_grpc.EmbeddingServicer):
    def Embed(
        self, request: embeddings_pb.EmbeddingRequest, _: grpc.RpcContext
    ) -> embeddings_pb.BatchEmbeds:
        embedding_model = em.SentenceTransformerEmbeddings(
            em.SentenceTransformerEmbeddingsConfig(
                model_name=request.model_name,
                batch_size=request.batch_size,
            )
        )

        embeds = embedding_model.embedding_fn()(list(request.strings))

        embeds_pb = [embeddings_pb.Embed(embed=e) for e in embeds]

        return embeddings_pb.BatchEmbeds(embeds=embeds_pb)


class RemoteEmbeddingsConfig(em.SentenceTransformerEmbeddingsConfig):
    api_base: str = "localhost"
    port: int = 50052


class RemoteEmbeddings(em.SentenceTransformerEmbeddings):
    def __init__(self, config: RemoteEmbeddingsConfig = RemoteEmbeddingsConfig()):
        super().__init__(config)
        self.config: RemoteEmbeddingsConfig = config

    def embedding_fn(self) -> Callable[[list[str]], lr.mytypes.Embeddings]:
        def fn(texts: list[str]) -> lr.mytypes.Embeddings:
            url = f"{self.config.api_base}:{self.config.port}"
            print(f"Attempting to connect to {url}")
            with grpc.insecure_channel(url) as channel:
                stub = embeddings_grpc.EmbeddingStub(channel)  # type: ignore
                response = stub.Embed(
                    embeddings_pb.EmbeddingRequest(
                        model_name=self.config.model_name,
                        batch_size=self.config.batch_size,
                        strings=texts,
                    )
                )

                return [list(emb.embed) for emb in response.embeds]

        def with_handling(texts: list[str]) -> lr.mytypes.Embeddings:
            try:
                return fn(texts)
            # Occurs when the server hasn't been started
            except grpc.RpcError:
                # Start the server
                subprocess.run(
                    ["python3", __file__, self.config.api_base, str(self.config.port)]
                )
                return fn(texts)

        return with_handling


def serve(
    bind_address: str = "localhost", port: int = 50052, max_workers: int = 10
) -> None:
    """Starts the RPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    embeddings_grpc.add_EmbeddingServicer_to_server(RemoteEmbeddingRPCs(), server)  # type: ignore
    server.add_insecure_port(f"{bind_address}:{port}")
    server.start()
    print(f"Embedding server started, listening on localhost:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    Fire(serve)
