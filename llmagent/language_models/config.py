from dataclasses import dataclass

@dataclass
class LLMConfig:
    type: str = "openai"
    max_tokens:int = 1024


