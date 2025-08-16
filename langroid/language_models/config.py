from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptFormatterConfig(BaseSettings):
    type: str = "llama2"

    model_config = SettingsConfigDict(env_prefix="FORMAT_", case_sensitive=False)


class Llama2FormatterConfig(PromptFormatterConfig):
    use_bos_eos: bool = False


class HFPromptFormatterConfig(PromptFormatterConfig):
    type: str = "hf"
    model_name: str
