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


# define a retry decorator
def retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (  # type: ignore
        requests.exceptions.RequestException,
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.APIError,
        openai.error.ServiceUnavailableError,
        openai.error.TryAgain,
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except openai.error.InvalidRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warning(
                    f"""OpenAI API request failed with error: 
                    {e}. 
                    Retrying in {delay} seconds..."""
                )
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def async_retry_with_exponential_backoff(
    func: Callable[..., Any],
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (  # type: ignore
        openai.error.Timeout,
        openai.error.RateLimitError,
        openai.error.APIError,
        openai.error.ServiceUnavailableError,
        openai.error.TryAgain,
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
    ),
) -> Callable[..., Any]:
    """Retry a function with exponential backoff."""

    async def wrapper(*args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or exception is raised
        while True:
            try:
                result = await func(*args, **kwargs)
                return result

            except openai.error.InvalidRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warning(
                    f"""OpenAI API request failed with error{e}. 
                    Retrying in {delay} seconds..."""
                )
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


# @retry_with_exponential_backoff
# def completions_with_backoff(**kwargs):
#     return openai.Completion.create(**kwargs)


# completions_with_backoff(model="text-davinci-002", prompt="Once upon a time,")
