from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dialog
    from . import prompts_config
    from . import templates
    from . import transforms
else:
    dialog = LazyLoad("langroid.prompts.dialog")
    prompts_config = LazyLoad("langroid.prompts.prompts_config")
    templates = LazyLoad("langroid.prompts.templates")
    transforms = LazyLoad("langroid.prompts.transforms")

__all__ = [
    "dialog",
    "prompts_config",
    "templates",
    "transforms",
]
