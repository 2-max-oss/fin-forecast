"""Exponential backoff retry decorator."""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

from config import RETRY_INITIAL_DELAY, RETRY_MAX, RETRY_MAX_DELAY
from core.exceptions import DataFetchError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def with_retry(
    max_retries: int = RETRY_MAX,
    initial_delay: float = RETRY_INITIAL_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry *fn* up to *max_retries* times with exponential backoff."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        raise DataFetchError(
                            f"{fn.__name__} failed after {max_retries} retries: {exc}"
                        ) from exc
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        fn.__name__, attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
        return wrapper  # type: ignore[return-value]
    return decorator
