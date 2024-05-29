from langroid.utils.system import LazyLoad
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import configuration
    from . import globals
    from . import constants
    from . import logging
    from . import pydantic_utils
    from . import system
    from . import output
else:
    configuration = LazyLoad("langroid.utils.configuration")
    globals = LazyLoad("langroid.utils.globals")
    constants = LazyLoad("langroid.utils.constants")
    logging = LazyLoad("langroid.utils.logging")
    pydantic_utils = LazyLoad("langroid.utils.pydantic_utils")
    system = LazyLoad("langroid.utils.system")
    output = LazyLoad("langroid.utils.output")

__all__ = [
    "configuration",
    "globals",
    "constants",
    "logging",
    "pydantic_utils",
    "system",
    "output",
]
