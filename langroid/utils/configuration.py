import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseSettings


class Settings(BaseSettings):
    # NOTE all of these can be overridden in your .env file with upper-case names,
    # for example CACHE_TYPE=momento
    debug: bool = False  # show debug messages?
    progress: bool = False  # show progress spinners/bars?
    stream: bool = True  # stream output?
    cache: bool = True  # use cache?
    cache_type: str = "redis"  # cache type: "redis" or "momento"
    interactive: bool = True  # interactive mode?
    gpt3_5: bool = True  # use GPT-3.5?
    nofunc: bool = False  # use model without function_call? (i.e. gpt-4)
    chat_model: str = ""  # language model name, e.g. litellm/ollama/llama2

    class Config:
        extra = "forbid"


load_dotenv()  # get settings from .env file
settings = Settings()


def update_global_settings(cfg: BaseSettings, keys: List[str]) -> None:
    """
    Update global settings so modules can access them via (as an example):
    ```
    from langroid.utils.configuration import settings
    if settings.debug...
    ```
    Caution we do not want to have too many such global settings!
    Args:
        cfg: pydantic config, typically from a main script
        keys: which keys from cfg to use, to update the global settings object
    """
    config_dict = cfg.dict()

    # Filter the config_dict based on the keys
    filtered_config = {key: config_dict[key] for key in keys if key in config_dict}

    # create a new Settings() object to let pydantic validate it
    new_settings = Settings(**filtered_config)

    # Update the unique global settings object
    settings.__dict__.update(new_settings.__dict__)


def set_global(key_vals: Settings) -> None:
    """Update the unique global settings object"""
    settings.__dict__.update(key_vals.__dict__)


def set_env(settings: BaseSettings) -> None:
    """
    Set environment variables from a BaseSettings instance
    Args:
        settings (BaseSettings): desired settings
    Returns:
    """
    for field_name, field in settings.__class__.__fields__.items():
        env_var_name = field.field_info.extra.get("env", field_name).upper()
        os.environ[env_var_name] = str(settings.dict()[field_name])
