from dataclasses import dataclass


@dataclass
class PromptsConfig:
    max_tokens: int = 1000
