from pydantic import BaseModel

class EmbeddingModelsConfig(BaseModel):
    model_type: str = "openai"



