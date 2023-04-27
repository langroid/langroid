from dataclasses import dataclass

@dataclass
class ParsingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    