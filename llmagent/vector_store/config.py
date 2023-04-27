from dataclasses import dataclass

@dataclass
class VectorStoreConfig:
    type: str = "qdrant"
    compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"


