from _typeshed import Incomplete

class EmbeddingStub:
    Embed: Incomplete
    def __init__(self, channel) -> None: ...

class EmbeddingServicer:
    def Embed(self, request, context) -> None: ...

def add_EmbeddingServicer_to_server(servicer, server) -> None: ...

class Embedding:
    @staticmethod
    def Embed(
        request,
        target,
        options=(),
        channel_credentials: Incomplete | None = None,
        call_credentials: Incomplete | None = None,
        insecure: bool = False,
        compression: Incomplete | None = None,
        wait_for_ready: Incomplete | None = None,
        timeout: Incomplete | None = None,
        metadata: Incomplete | None = None,
    ): ...
