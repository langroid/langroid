from pydantic import BaseSettings


class PromptsConfig(BaseSettings):
    max_tokens: int = 1000  # for output; NOT USED ANYWHERE
