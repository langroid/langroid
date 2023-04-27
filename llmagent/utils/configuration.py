# will be set by main script at run-time
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel
from typing import List

class Settings(BaseModel):
    debug: bool = False
    class Config:
        extra = "forbid"

settings = Settings()

def update_global_settings(cfg: DictConfig, keys: List[str]) -> None:
    """
    Update global settings so modules can access them via (as an example):
    ```
    from llmagent.utils.configuration import settings
    if settings.debug...
    ```
    Caution we do not want to have too many such global settings!
    Args:
        cfg: DictConfig, typically in a main script via Hydra
        keys: which keys from cfg to use, to update the global settings object
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Filter the config_dict based on the keys
    filtered_config = {key: config_dict[key] for key in keys if key in config_dict}

    # create a new Settings() object to let pydantic validate it
    new_settings = Settings(**filtered_config)

    # Update the unique global settings object
    settings.__dict__.update(new_settings.__dict__)




