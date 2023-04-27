from pydantic import BaseModel

class VectorStoreConfig(BaseModel):
    type: str = "qdrant"
    compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"


