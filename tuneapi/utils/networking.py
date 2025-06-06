# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License
# REMEMBER: nothing from outside tune should be imported in utils

import time
from typing import Any, Dict

from tuneapi.utils.logger import logger


class UnAuthException(Exception):
    """Raised when the API returns a 401"""


class DoNotRetryException(Exception):
    """Raised when code tells not to retry"""


def exponential_backoff(
    foo, *args, max_retries=2, retry_delay=1, **kwargs
) -> Dict[str, Any]:
    """Exponential backoff function

    Args:
        foo (function): The function to call
        max_retries (int, optional): maximum number of retries. Defaults to 2.
        retry_delay (int, optional): Initial delay in seconds. Defaults to 1.

    Raises:
        e: Max retries reached. Exiting...
        Exception: This should never happen

    Returns:
        Dict[str, Any]: The completion(s) generated by the API.
    """

    if not max_retries:
        try:
            out = foo(*args, **kwargs)  # Call the function that may crash
            return out  # If successful, break out of the loop and return
        except DoNotRetryException as e:
            raise e
        except UnAuthException as e:
            raise e
        except Exception as e:
            logger.warning(f"Function crashed: {e}")
            raise e

    for attempt in range(max_retries):
        try:
            out = foo(*args, **kwargs)  # Call the function that may crash
            return out  # If successful, break out of the loop and return
        except DoNotRetryException as e:
            raise e
        except UnAuthException as e:
            raise e
        except Exception as e:
            logger.warning(f"Function crashed: {e}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Exiting...")
                raise e
            else:
                delay = retry_delay * (2**attempt)  # Calculate the backoff delay
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait for the calculated delay

    raise Exception("This should never happen")
