from pydantic import BaseModel

class ParsingConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    