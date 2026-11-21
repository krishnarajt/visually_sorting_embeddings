import logging
import logging.handlers
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings


def setup_logging():
    """Configure application-wide logging"""
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.TimedRotatingFileHandler(
                "logs/app.log",
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            ),
        ],
    )

    for noisy in ("httpx", "httpcore", "uvicorn.access", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # chromadb 0.6.3 ships a telemetry call whose signature does not match its
    # own capture(), so it logs a failure for every operation even with
    # anonymized_telemetry=False. Nothing is being sent; the noise is the only
    # symptom, and it would otherwise be one line per request.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {settings.log_level} level")
    return logger


logger = setup_logging()

from app.api.routes import router as api_router  # noqa: E402
from app.ratelimit import RateLimitMiddleware  # noqa: E402
from app.services.embedding import (  # noqa: E402
    NATIVE_SOURCES,
    PIVOT_SOURCES,
    glove,
    source_status,
)
from app.services.projection import umap_available  # noqa: E402


def warm_umap():
    """Run one throwaway UMAP fit so the first real request is not the slow one.

    UMAP is numba-compiled and its first call in a process spends about twelve
    seconds in the JIT before doing any work; every call after that is thirty
    milliseconds. Since UMAP is the default method, leaving that cost in place
    means the first thing a visitor experiences is a twelve-second hang that
    looks exactly like a broken server. Paying it here, on a background thread
    during startup, makes it invisible.
    """
    try:
        import numpy as np
        import umap

        rng = np.random.default_rng(0)
        umap.UMAP(n_components=3, n_neighbors=4, metric="cosine").fit_transform(
            rng.normal(size=(12, 16))
        )
        logger.info("✓ UMAP warmed up")
    except Exception as exc:
        logger.warning(f"⚠ UMAP warm-up failed, first projection will be slow: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting visually sorting embeddings...")
    logger.info("=" * 60)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Vector store at {settings.chroma_dir}")

    # Only the native sources are listed. Each has a translate-to-English twin
    # whose availability is by definition its own, and printing sixteen lines
    # to say eight things would bury the ones that matter.
    for source in NATIVE_SOURCES:
        ok, why = source_status(source)
        if ok:
            logger.info(f"✓ {source.label} ready ({source.dims}d)")
        else:
            # Not fatal. A missing API key leaves GloVe and the local models,
            # a missing GloVe file leaves the API ones, and unfetched weights
            # leave both; the UI greys out what it cannot use and says why, so
            # a half-configured server is still a usable one.
            logger.warning(f"⚠ {source.label} unavailable — {why}")
    ready = sum(1 for s in PIVOT_SOURCES if source_status(s)[0])
    logger.info(f"✓ {ready} translate-to-English twins, one per available model")

    # The ONNX encoders are deliberately *not* warmed the way UMAP is. UMAP's
    # first call wastes twelve seconds on a JIT that every later call reuses,
    # so paying it at startup is free. These three are 2.8 GB of weights, and a
    # visitor who only ever looks at GloVe should not be made to hold LaBSE in
    # memory to do it. The cost is that the first request to one of them spends
    # a few seconds mapping the graph in.

    if glove.available:
        # Reading 55 MB off disk on the first request would show up as lag on
        # whichever unlucky word triggered it.
        threading.Thread(target=glove.load, name="glove-load", daemon=True).start()

    if umap_available():
        threading.Thread(target=warm_umap, name="umap-warm", daemon=True).start()
    else:
        logger.warning("⚠ umap-learn not installed — five of six methods available")

    logger.info("=" * 60)
    logger.info(f"Ready on port {settings.port}. Go and look at some words.")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="visually sorting embeddings",
    description=(
        "Embeddings have hundreds or thousands of dimensions and screens have "
        "three. This serves the same words flattened six different ways, with "
        "the true cosine similarities alongside, so you can see how much each "
        "way of looking distorts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Starlette runs middleware in reverse order of registration, so the limiter
# goes first to end up *inside* CORS: its 429 then carries CORS headers and
# the browser can read it. See app/ratelimit.py.
if settings.rate_limit_per_minute > 0:
    app.add_middleware(
        RateLimitMiddleware, per_minute=settings.rate_limit_per_minute
    )
    logger.info(f"Rate limit: {settings.rate_limit_per_minute} requests/min per client")

# CORS_ORIGINS=* is for the deployed API. There are no accounts, cookies or
# Authorization headers here, so there is nothing a credentialed request could
# reach that an anonymous one cannot — and the alternative is listing every
# Vercel and Netlify deploy URL, preview deployments included, which are
# different on every pull request. A wildcard cannot be combined with
# credentials, so they are switched off in that mode; named origins keep them
# on, as before.
allow_any_origin = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_any_origin else settings.cors_origins,
    allow_credentials=not allow_any_origin,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(
    "CORS enabled for any origin"
    if allow_any_origin
    else f"CORS enabled for origins: {settings.cors_origins}"
)

app.include_router(api_router, prefix="/api")
logger.info("API routers registered")


@app.get("/")
def root():
    return {
        "name": "visually sorting embeddings",
        "description": "high-dimensional embeddings, flattened six ways",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health_check():
    """Health check endpoint for k8s"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
def readiness_check():
    """Readiness check endpoint for k8s"""
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {settings.port}, reload={settings.reload}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
