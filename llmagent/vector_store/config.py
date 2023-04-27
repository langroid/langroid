from dataclasses import dataclass

@dataclass
class VectorStoreConfig:
    collection_name: str = "default"
    storage_path: str = ".qdrant/data",
    type: str = "qdrant"
    embedding_fn_type: str = "openai"
    host: str = "127.0.0.1"
    port: int = 6333
    #compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"


