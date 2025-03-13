# from openai-cookbook
import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, List

import aiohttp
import openai
import requests

logger = logging.getLogger(__name__)
# setlevel to warning
logger.setLevel(logging.WARNING)


base_retry_errors = (
    requests.exceptions.RequestException,
    aiohttp.ServerTimeoutError,
    asyncio.TimeoutError,
)


# define a retry decorator
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    retryable_errors: tuple = base_retry_errors  # type: ignore
    + (  # type: ignore
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIError,
    ),
    terminal_errors: tuple = (  # type: ignore
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.UnprocessableEntityError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries
            # is hit or exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                except terminal_errors as e:
                    logger.error(f"API request failed with terminal error: {e}.")
                    raise e

                # Retry on specified errors
                except retryable_errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                            f" Last error: {str(e)}."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    logger.warning(
                        f"""API request failed with error: 
                        {e}. 
                        Retrying in {delay} seconds..."""
                    )
                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    retryable_errors: tuple = base_retry_errors  # type: ignore
    + (  # type: ignore
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIError,
    ),
    terminal_errors: tuple = (  # type: ignore
        openai.BadRequestError,
        openai.AuthenticationError,
        openai.UnprocessableEntityError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    async def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries
            # is hit or exception is raised
            while True:
                try:
                    result = await func(*args, **kwargs)
                    return result

                # Do not retry for terminal tuple
                except terminal_errors as terminal_error:
                    logger.error(f"API request failed with error: {terminal_error}.")
                    raise terminal_error

                # Retry on specified errors
                except retryable_errors as retryable_error:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                            f" Last error: {str(retryable_error)}."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    logger.warning(
                        f"""API request failed with error{retryable_error}. 
                        Retrying in {delay} seconds..."""
                    )
                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator
