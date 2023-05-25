from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    debug: bool = False  # show debug messages?
    progress: bool = False  # show progress spinners/bars?
    stream: bool = True  # stream output?
    cache: bool = True  # use cache?
    gpt3_5: bool = True  # use GPT-3.5?

    class Config:
        extra = "forbid"


settings = Settings()


def update_global_settings(cfg: BaseSettings, keys: List[str]) -> None:
    """
    Update global settings so modules can access them via (as an example):
    ```
    from llmagent.utils.configuration import settings
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
    # Update the unique global settings object
    settings.__dict__.update(key_vals.__dict__)
