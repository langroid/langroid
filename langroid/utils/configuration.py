import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Literal, cast

from dotenv import find_dotenv, load_dotenv

from langroid.pydantic_v1 import BaseSettings

# Global reentrant lock to serialize any modifications to the global settings.
_global_lock = threading.RLock()


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


# Load environment variables from .env file.
load_dotenv(find_dotenv(usecwd=True))

# The global (default) settings instance.
# This is updated by update_global_settings() and set_global().
_global_settings = Settings()

# Thread-local storage for temporary (per-thread) settings overrides.
_thread_local = threading.local()


class SettingsProxy:
    """
    A proxy for the settings that returns a thread‐local override if set,
    or else falls back to the global settings.
    """

    def __getattr__(self, name: str) -> Any:
        # If the calling thread has set an override, use that.
        if hasattr(_thread_local, "override"):
            return getattr(_thread_local.override, name)
        return getattr(_global_settings, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # All writes go to the global settings.
        setattr(_global_settings, name, value)

    def update(self, new_settings: Settings) -> None:
        _global_settings.__dict__.update(new_settings.__dict__)

    def dict(self) -> Dict[str, Any]:
        # Return a dict view of the settings as seen by the caller.
        # Note that temporary overrides are not “merged” with global settings.
        if hasattr(_thread_local, "override"):
            return cast(Dict[str, Any], cast(Settings, _thread_local.override.dict()))
        return _global_settings.dict()


settings = SettingsProxy()


def update_global_settings(cfg: BaseSettings, keys: List[str]) -> None:
    """
    Update global settings so that modules can later access them via, e.g.,

        from langroid.utils.configuration import settings
        if settings.debug: ...

    This updates the global default.
    """
    config_dict = cfg.dict()
    filtered_config = {key: config_dict[key] for key in keys if key in config_dict}
    new_settings = Settings(**filtered_config)
    _global_settings.__dict__.update(new_settings.__dict__)


def set_global(key_vals: Settings) -> None:
    """
    Update the global settings object.
    """
    _global_settings.__dict__.update(key_vals.__dict__)


@contextmanager
def temporary_settings(temp_settings: Settings) -> Iterator[None]:
    """
    Temporarily override the settings for the calling thread.

    Within the context, any access to "settings" will use the provided temporary
    settings. Once the context is exited, the thread reverts to the global settings.
    """
    saved = getattr(_thread_local, "override", None)
    _thread_local.override = temp_settings
    try:
        yield
    finally:
        if saved is not None:
            _thread_local.override = saved
        else:
            del _thread_local.override


@contextmanager
def quiet_mode(quiet: bool = True) -> Iterator[None]:
    """
    Temporarily override settings.quiet for the current thread.
    This implementation builds on the thread‑local temporary_settings context manager.
    The effective quiet mode is merged:
    if quiet is already True (from an outer context),
    then it remains True even if a nested context passes quiet=False.
    """
    current_effective = settings.dict()  # get the current thread's effective settings
    # Create a new settings instance from the current effective state.
    temp = Settings(**current_effective)
    # Merge the new flag: once quiet is enabled, it stays enabled.
    temp.quiet = settings.quiet or quiet
    with temporary_settings(temp):
        yield


def set_env(settings_instance: BaseSettings) -> None:
    """
    Set environment variables from a BaseSettings instance.

    Each field in the settings is written to os.environ.
    """
    for field_name, field in settings_instance.__class__.__fields__.items():
        env_var_name = field.field_info.extra.get("env", field_name).upper()
        os.environ[env_var_name] = str(settings_instance.dict()[field_name])
