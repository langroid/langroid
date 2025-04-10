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
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (  # type: ignore
        requests.exceptions.RequestException,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.AuthenticationError,
        openai.APIError,
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

            except openai.BadRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e
            except openai.AuthenticationError as e:
                # do not retry when there's an auth error
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e

            except openai.UnprocessableEntityError as e:
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e

            # Retry on specified errors
            except errors as e:

                # For certain types of errors that slip through here
                # (e.g. when using proxies like LiteLLM, do not retry)
                if any(
                    err in str(e)
                    for err in [
                        "BadRequestError",
                        "ConnectionError",
                    ]
                ):
                    logger.error(f"OpenAI API request failed with error: {e}.")
                    raise e
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
    exponential_base: float = 1.3,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (  # type: ignore
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.AuthenticationError,
        openai.APIError,
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

            except openai.BadRequestError as e:
                # do not retry when the request itself is invalid,
                # e.g. when context is too long
                logger.error(f"OpenAI API request failed with error: {e}.")
                raise e
            except openai.AuthenticationError as e:
                # do not retry when there's an auth error
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
                        f" Last error: {str(e)}."
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
