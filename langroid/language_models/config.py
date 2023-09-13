from pydantic import BaseSettings


class PromptFormatterConfig(BaseSettings):
    type: str = "llama2"

    class Config:
        env_prefix = "FORMAT_"
        case_sensitive = False


class Llama2FormatterConfig(PromptFormatterConfig):
    use_bos_eos: bool = False
