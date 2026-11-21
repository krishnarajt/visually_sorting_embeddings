# syntax=docker/dockerfile:1

# visually sorting embeddings — API
#
# The image carries everything the app reads at runtime, because the weights
# are never downloaded on demand (see app/services/encoders.py): GloVe, and the
# local ONNX encoders. That makes it large — about 1 GB of Python
# dependencies and GloVe, plus up to 5 GB of weights (6.1 GB built) — and the layout
# below exists so that is a cost paid once rather than on every push.
#
# Each asset is fetched in its own stage, from a layer that contains only the
# few files the fetch scripts import. Editing the app therefore does not
# invalidate any download, and each model ends up as its own registry layer
# instead of one 5 GB blob that has to upload in one piece.
#
# WHICH MODELS. All four local encoders are baked in. They load lazily on the
# first request that needs them and are never evicted, so an idle pod holds
# none of them — but one that has served all four holds about 5 GB, which is
# what the memory limit in k8s/base/deployment.yaml is sized for. To slim the
# image, delete the stage and the COPY line for a model at the bottom. The
# source then reports itself unavailable and the UI greys it out; nothing else
# changes. Sizes: bge-m3 2.2 GB, LaBSE 1.9 GB, e5-small 0.5 GB, MiniLM 0.5 GB.

# ── deps: the Python environment ──────────────────────────────
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


# ── fetch: deps plus only what the fetch scripts import ───────
FROM deps AS fetch

ENV DATA_DIR=/data \
    HF_HUB_DISABLE_TELEMETRY=1

COPY app/__init__.py app/__init__.py
COPY app/settings.py app/settings.py
COPY app/services/__init__.py app/services/__init__.py
COPY app/services/encoders.py app/services/encoders.py
COPY scripts/fetch_glove.py scripts/fetch_models.py scripts/


# ── GloVe: 820 MB zip in, 55 MB out, the zip never reaches the image ──
FROM fetch AS glove
RUN python scripts/fetch_glove.py


# ── The ONNX encoders, one stage each ─────────────────────────
# `.cache` is the hub's download bookkeeping and nothing reads it afterwards.
FROM fetch AS model-e5
RUN python scripts/fetch_models.py --model e5-multilingual-small \
    && rm -rf /data/models/*/.cache

FROM fetch AS model-minilm
RUN python scripts/fetch_models.py --model minilm-multilingual \
    && rm -rf /data/models/*/.cache

FROM fetch AS model-labse
RUN python scripts/fetch_models.py --model labse \
    && rm -rf /data/models/*/.cache

FROM fetch AS model-bge-m3
RUN python scripts/fetch_models.py --model bge-m3 \
    && rm -rf /data/models/*/.cache


# ── final: deps, then the assets, then the code ───────────────
# Code goes last because it is the only part that changes between pushes.
FROM deps AS final

# No .env in the image, on purpose: see .dockerignore. Keys arrive as
# environment variables from the cluster's secret.
ENV PORT=8000

COPY --from=glove /data/glove.6B.300d.npz /app/data/glove.6B.300d.npz
COPY --from=model-e5 /data/models/ /app/data/models/
COPY --from=model-minilm /data/models/ /app/data/models/
COPY --from=model-labse /data/models/ /app/data/models/
COPY --from=model-bge-m3 /data/models/ /app/data/models/

COPY main.py .
COPY app ./app

EXPOSE 8000

# One worker: the projection cache, the loaded encoders and the rate limiter
# all live in this process, and a second worker would hold a second copy of
# each — with the encoders that is another 5 GB.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-access-log"]
