"""A small retrying JSON fetcher shared by both source clients.

Isolated behind a callable so that clients take their transport as an
argument. Tests pass a fake and run offline; CI never depends on CDC or the
Census Bureau being reachable, which is the difference between a suite that
tells you about your code and one that tells you about the internet.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping

from src.ingest import config

logger = logging.getLogger(__name__)

Fetcher = Callable[[str, Mapping[str, Any]], Any]


class FetchError(RuntimeError):
    """Raised when a request fails after exhausting retries."""


class MissingCredentialError(RuntimeError):
    """Raised when a required API key is absent from the environment."""


def _is_retryable(status: int) -> bool:
    # 429 is rate limiting; 5xx is the server's problem, not the request's.
    # A 4xx other than 429 will fail identically on every retry, so retrying
    # it just delays a failure the caller needs to see.
    return status == 429 or 500 <= status < 600


def fetch_json(
    url: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout: int = config.REQUEST_TIMEOUT_SECONDS,
    max_retries: int = config.MAX_RETRIES,
    backoff: float = config.RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET a URL and parse the JSON body, retrying transient failures.

    Backoff is exponential. Public data APIs rate-limit unauthenticated
    callers, and a backfill issues enough requests to hit that ceiling.
    """
    import json

    query = urllib.parse.urlencode(dict(params or {}), safe="$,()'*")
    full_url = f"{url}?{query}" if query else url

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(
            full_url, headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if not _is_retryable(exc.code):
                body = exc.read()[:400].decode("utf-8", errors="replace")
                raise FetchError(
                    f"{exc.code} from {url} (not retryable): {body}"
                ) from exc
            logger.warning(
                "Attempt %s/%s: HTTP %s from %s", attempt, max_retries, exc.code, url
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            logger.warning("Attempt %s/%s: %s from %s", attempt, max_retries, exc, url)

        if attempt < max_retries:
            sleep(backoff * (2 ** (attempt - 1)))

    raise FetchError(
        f"Giving up on {url} after {max_retries} attempts: {last_error}"
    ) from last_error
