"""
If run as a script, starts an RPC server which handles remote
embedding requests:

For example:
python3 -m langroid.embedding_models.remote_embeds --port `port`

where `port` is the port at which the service is exposed.  Currently,
supports insecure connections only, and this should NOT be exposed to
the internet.
"""

import atexit
import subprocess
import time
from typing import Callable, Optional

import grpc
from fire import Fire

import langroid.embedding_models.models as em
import langroid.embedding_models.protoc.embeddings_pb2 as embeddings_pb
import langroid.embedding_models.protoc.embeddings_pb2_grpc as embeddings_grpc
from langroid.mytypes import Embeddings


class RemoteEmbeddingRPCs(embeddings_grpc.EmbeddingServicer):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        data_parallel: bool,
        device: Optional[str],
        devices: Optional[list[str]],
    ):
        super().__init__()
        self.embedding_fn = em.SentenceTransformerEmbeddings(
            em.SentenceTransformerEmbeddingsConfig(
                model_name=model_name,
                batch_size=batch_size,
                data_parallel=data_parallel,
                device=device,
                devices=devices,
            )
        ).embedding_fn()

    def Embed(
        self, request: embeddings_pb.EmbeddingRequest, _: grpc.RpcContext
    ) -> embeddings_pb.BatchEmbeds:
        embeds = self.embedding_fn(list(request.strings))

        embeds_pb = [embeddings_pb.Embed(embed=e) for e in embeds]

        return embeddings_pb.BatchEmbeds(embeds=embeds_pb)


class RemoteEmbeddingsConfig(em.SentenceTransformerEmbeddingsConfig):
    api_base: str = "localhost"
    port: int = 50052
    # The below are used only when waiting for server creation
    poll_delay: float = 0.01
    max_retries: int = 1000


class RemoteEmbeddings(em.SentenceTransformerEmbeddings):
    def __init__(self, config: RemoteEmbeddingsConfig = RemoteEmbeddingsConfig()):
        super().__init__(config)
        self.config: RemoteEmbeddingsConfig = config
        self.have_started_server: bool = False

    def embedding_fn(self) -> Callable[[list[str]], Embeddings]:
        def fn(texts: list[str]) -> Embeddings:
            url = f"{self.config.api_base}:{self.config.port}"
            with grpc.insecure_channel(url) as channel:
                stub = embeddings_grpc.EmbeddingStub(channel)  # type: ignore
                response = stub.Embed(
                    embeddings_pb.EmbeddingRequest(
                        strings=texts,
                    )
                )

                return [list(emb.embed) for emb in response.embeds]

        def with_handling(texts: list[str]) -> Embeddings:
            # In local mode, start the server if it has not already
            # been started
            if self.config.api_base == "localhost" and not self.have_started_server:
                try:
                    return fn(texts)
                # Occurs when the server hasn't been started
                except grpc.RpcError:
                    self.have_started_server = True
                    # Start the server
                    proc = subprocess.Popen(
                        [
                            "python3",
                            __file__,
                            "--bind_address_base",
                            self.config.api_base,
                            "--port",
                            str(self.config.port),
                            "--batch_size",
                            str(self.config.batch_size),
                            "--model_name",
                            self.config.model_name,
                        ],
                    )

                    atexit.register(lambda: proc.terminate())

                    for _ in range(self.config.max_retries - 1):
                        try:
                            return fn(texts)
                        except grpc.RpcError:
                            time.sleep(self.config.poll_delay)

            # The remote is not local or we have exhausted retries
            # We should now raise an error if the server is not accessible
            return fn(texts)

        return with_handling


async def serve(
    bind_address_base: str = "localhost",
    port: int = 50052,
    batch_size: int = 512,
    data_parallel: bool = False,
    device: Optional[str] = None,
    devices: Optional[list[str]] = None,
    model_name: str = "BAAI/bge-large-en-v1.5",
) -> None:
    """Starts the RPC server."""
    server = grpc.aio.server()
    embeddings_grpc.add_EmbeddingServicer_to_server(
        RemoteEmbeddingRPCs(
            model_name=model_name,
            batch_size=batch_size,
            data_parallel=data_parallel,
            device=device,
            devices=devices,
        ),
        server,
    )  # type: ignore
    url = f"{bind_address_base}:{port}"
    server.add_insecure_port(url)
    await server.start()
    print(f"Embedding server started, listening on {url}")
    await server.wait_for_termination()


if __name__ == "__main__":
    Fire(serve)
