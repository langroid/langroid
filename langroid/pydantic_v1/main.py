try:
    from pydantic.v1.main import *  # noqa: F403, F401
except ImportError:
    from pydantic.main import *  # type: ignore # noqa: F403, F401
