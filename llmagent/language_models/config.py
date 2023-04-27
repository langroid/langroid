from pydantic import BaseModel

class LLMConfig(BaseModel):
    type: str = "openai"


