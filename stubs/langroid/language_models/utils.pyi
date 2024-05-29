from typing import Any, Callable

from _typeshed import Incomplete

logger: Incomplete

def retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = ...,
) -> Callable[..., Any]: ...
def async_retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = ...,
) -> Callable[..., Any]: ...
