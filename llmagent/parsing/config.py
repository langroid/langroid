from dataclasses import dataclass, field
from typing import List

@dataclass
class ParsingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    token_encoding_model: str = "text-davinci-003"
    