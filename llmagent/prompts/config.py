from pydantic import BaseModel

class PromptsConfig(BaseModel):
    max_tokens:int = 1000


