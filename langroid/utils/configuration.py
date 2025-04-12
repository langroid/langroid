import copy
import os
import threading
from contextlib import contextmanager
from typing import Iterator, List, Literal

from dotenv import find_dotenv, load_dotenv

from langroid.pydantic_v1 import BaseSettings

# Global reentrant lock to ensure thread-safe updates to "settings"
_settings_lock = threading.RLock()


class Settings(BaseSettings):
    debug: bool = False  # show debug messages?
    max_turns: int = -1  # maximum number of turns in a task (to avoid inf loop)
    progress: bool = False  # show progress spinners/bars?
    stream: bool = True  # stream output?
    cache: bool = True  # use cache?
    cache_type: Literal["redis", "fakeredis", "momento", "none"] = "redis"  # cache type
    chat_model: str = ""  # language model name, e.g. litellm/ollama/llama2
    quiet: bool = False  # quiet mode (i.e. suppress all output)?
    notebook: bool = False  # running in a notebook?

    class Config:
        extra = "forbid"


load_dotenv(find_dotenv(usecwd=True))  # get settings from .env file
settings = Settings()


def update_global_settings(cfg: BaseSettings, keys: List[str]) -> None:
    """
    Update global settings so modules can access them via (as an example):

        from langroid.utils.configuration import settings
        if settings.debug: ...

    Caution: We do not want to have too many global settings!
    Args:
        cfg: pydantic config, typically from a main script
        keys: which keys from cfg to use, to update the global settings object
    """
    config_dict = cfg.dict()
    # Filter the config_dict based on the keys
    filtered_config = {key: config_dict[key] for key in keys if key in config_dict}

    # Create a new Settings instance so pydantic validates the config
    new_settings = Settings(**filtered_config)

    # Atomically update the unique global settings object
    with _settings_lock:
        settings.__dict__.update(new_settings.__dict__)


def set_global(key_vals: Settings) -> None:
    """Atomically update the unique global settings object"""
    with _settings_lock:
        settings.__dict__.update(key_vals.__dict__)


@contextmanager
def temporary_settings(temp_settings: Settings) -> Iterator[None]:
    """Temporarily update the global settings and restore them afterward."""
    with _settings_lock:
        original_settings = copy.deepcopy(settings)
        set_global(temp_settings)
    try:
        yield
    finally:
        with _settings_lock:
            settings.__dict__.update(original_settings.__dict__)


@contextmanager
def quiet_mode(quiet: bool = True) -> Iterator[None]:
    """Temporarily set quiet=True in global settings and restore afterward."""
    with _settings_lock:
        original_settings = copy.deepcopy(settings)
        if quiet:
            # Use Pydantic's copy/update method to create a modified instance
            temp_settings = original_settings.copy(update={"quiet": True})
            set_global(temp_settings)
    try:
        yield
    finally:
        with _settings_lock:
            settings.__dict__.update(original_settings.__dict__)


def set_env(settings_instance: BaseSettings) -> None:
    """
    Set environment variables from a BaseSettings instance.
    Args:
        settings_instance (BaseSettings): desired settings.
    """
    # Using the lock ensures a consistent read if the passed settings
    # happen to be the global one.
    with _settings_lock:
        for field_name, field in settings_instance.__class__.__fields__.items():
            env_var_name = field.field_info.extra.get("env", field_name).upper()
            os.environ[env_var_name] = str(settings_instance.dict()[field_name])
