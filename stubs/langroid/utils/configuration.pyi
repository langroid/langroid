from typing import Iterator, Literal

from _typeshed import Incomplete
from pydantic import BaseSettings

class Settings(BaseSettings):
    debug: bool
    max_turns: int
    progress: bool
    stream: bool
    cache: bool
    cache_type: Literal["redis", "fakeredis", "momento"]
    interactive: bool
    gpt3_5: bool
    chat_model: str
    quiet: bool
    notebook: bool

    class Config:
        extra: str

settings: Incomplete

def update_global_settings(cfg: BaseSettings, keys: list[str]) -> None: ...
def set_global(key_vals: Settings) -> None: ...
def temporary_settings(temp_settings: Settings) -> Iterator[None]: ...
def quiet_mode(quiet: bool = True) -> Iterator[None]: ...
def set_env(settings: BaseSettings) -> None: ...
