from pydantic import BaseSettings

class PromptFormatterConfig(BaseSettings):
    type: str

    class Config:
        env_prefix: str
        case_sensitive: bool

class Llama2FormatterConfig(PromptFormatterConfig):
    use_bos_eos: bool

class HFPromptFormatterConfig(PromptFormatterConfig):
    type: str
    model_name: str
