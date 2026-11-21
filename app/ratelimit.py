"""A per-client request limiter, for when this API is on the public internet.

There is no login here, which is the point, and it means nothing but this
stands between a stranger and three things that cost something: the OpenAI and
Cohere keys behind `/api/scene` (every unseen word is a billable embedding), a
Cohere trial key whose monthly quota one script could exhaust for everybody,
and the CPU of a pod that is idle nearly all of the time and sized for it —
a 2000-word t-SNE is seconds of real work.

So: N requests per rolling minute per client, over `/api/*` only. It is
deliberately the smallest thing that does the job —

  * in-process, with no dependency and no Redis. The deployment is one replica
    with one worker, so one process *is* the whole picture; with more of
    either the limit would silently become N per process.
  * keyed on `CF-Connecting-IP` when present. Behind a tunnel every request
    arrives from the same internal address, and keying on that would make the
    whole internet share one bucket. The header is spoofable by anyone who can
    reach the pod directly, which only costs the limiter, not the data.
  * off when the limit is 0, which is the default, so local development and
    the test scripts never meet it.

It must sit *inside* the CORS middleware: a 429 without CORS headers reaches
the browser as an opaque network error, and the UI would show "Failed to
fetch" instead of the sentence this returns.
"""

import math
import time
from collections import deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

WINDOW_SECONDS = 60.0

# How often idle clients are forgotten. Without this the table only grows.
SWEEP_SECONDS = 60.0


def client_key(scope: Scope) -> str:
    """Who this request is from, as best the proxy chain lets us tell."""
    for name, value in scope.get("headers", ()):
        if name == b"cf-connecting-ip":
            return value.decode("latin-1").strip() or "unknown"
    client = scope.get("client")
    return client[0] if client else "unknown"


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, per_minute: int, prefix: str = "/api/") -> None:
        self.app = app
        self.limit = per_minute
        self.prefix = prefix
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = time.monotonic()

    def retry_after(self, key: str, now: float) -> int | None:
        """Record a hit; None if allowed, else the seconds until it would be."""
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= self.limit:
            # Refused requests are not recorded, so hammering the endpoint
            # cannot extend the wait past the oldest hit leaving the window.
            return max(1, math.ceil(WINDOW_SECONDS - (now - hits[0])))
        hits.append(now)
        return None

    def sweep(self, now: float) -> None:
        if now - self._last_sweep < SWEEP_SECONDS:
            return
        self._last_sweep = now
        for key in [
            k
            for k, hits in self._hits.items()
            if not hits or now - hits[-1] >= WINDOW_SECONDS
        ]:
            del self._hits[key]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # OPTIONS is the CORS preflight. The browser sends one before nearly
        # every real call here, and counting it would halve the allowance for
        # a request that does no work.
        if (
            self.limit <= 0
            or scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or not scope["path"].startswith(self.prefix)
        ):
            await self.app(scope, receive, send)
            return

        now = time.monotonic()
        self.sweep(now)
        wait = self.retry_after(client_key(scope), now)
        if wait is None:
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {
                "detail": (
                    f"Too many requests. This is a small shared server — "
                    f"try again in {wait} seconds."
                )
            },
            status_code=429,
            headers={"Retry-After": str(wait)},
        )
        await response(scope, receive, send)
