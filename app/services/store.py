"""The vector store: local Chroma, one collection per embedding source.

Chroma is doing two jobs here. The obvious one is persistence, so that
re-opening the page does not re-embed and re-pay for words already seen. The
less obvious one is that it enforces the separation between sources —
collections have a fixed dimensionality, so a 300-d GloVe vector physically
cannot be written into the 1536-d OpenAI collection. The invariant that makes
this app correct is held by the database rather than by care.

Collections are named `emb_<source id with hyphens as underscores>`: Chroma
requires 3-63 characters of alphanumerics, underscores and hyphens.

Ids come from `key_for`, not from a global normaliser, because case folding is
a property of the model — GloVe's vocabulary is lowercase and the transformer
models are not. Two sources disagreeing about the id for the same string is
harmless precisely because of the separation above: they never share a
collection.
"""

import logging
from datetime import datetime, timezone

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.services.embedding import Source, key_for
from app.settings import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None


def client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        logger.info(f"✓ Chroma opened at {settings.chroma_dir}")
    return _client


def collection_name(source: Source) -> str:
    return f"emb_{source.id.replace('-', '_')}"


def collection(source: Source) -> chromadb.Collection:
    """The collection for one source.

    `embedding_function=None` is deliberate and load-bearing. Chroma's default
    embedding function downloads an ONNX MiniLM model on first use and would
    embed our text itself — which is the one thing this app must never do,
    because then the vectors on screen would not be the model's under test.
    Every write here passes explicit vectors.
    """
    return client().get_or_create_collection(
        name=collection_name(source),
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(source: Source, texts: list[str], vectors: np.ndarray) -> None:
    if not texts:
        return
    if vectors.shape[0] != len(texts):
        raise ValueError(f"{len(texts)} texts but {vectors.shape[0]} vectors")
    now = datetime.now(timezone.utc).isoformat()
    collection(source).upsert(
        ids=[key_for(source, t) for t in texts],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"text": key_for(source, t), "source": source.id, "added_at": now}
            for t in texts
        ],
    )
    logger.info(f"Upserted {len(texts)} vectors into {collection_name(source)}")


def fetch(source: Source, texts: list[str]) -> dict[str, np.ndarray]:
    """Vectors already stored for these texts. Missing ones are simply absent."""
    if not texts:
        return {}
    ids = [key_for(source, t) for t in texts]
    got = collection(source).get(ids=ids, include=["embeddings"])
    # Chroma returns only the ids it has, in its own order, so zip rather than
    # assume alignment with the request.
    embeddings = got.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return {}
    return {
        str(i): np.asarray(e, dtype=np.float32)
        for i, e in zip(got["ids"], embeddings)
    }


def fetch_all(source: Source) -> tuple[list[str], np.ndarray]:
    """Every word in a source's index, with its vectors.

    This is the candidate pool for nearest-neighbour and arithmetic queries on
    the OpenAI sources, where there is no wider vocabulary to fall back on.
    """
    got = collection(source).get(include=["embeddings"])
    ids = [str(i) for i in got.get("ids", [])]
    embeddings = got.get("embeddings")
    if not ids or embeddings is None or len(embeddings) == 0:
        return [], np.zeros((0, source.dims), dtype=np.float32)
    return ids, np.asarray(embeddings, dtype=np.float32)


def words(source: Source) -> list[str]:
    got = collection(source).get(include=[])
    return sorted(str(i) for i in got.get("ids", []))


def count(source: Source) -> int:
    return collection(source).count()


def delete(source: Source, texts: list[str]) -> int:
    if not texts:
        return 0
    ids = [key_for(source, t) for t in texts]
    collection(source).delete(ids=ids)
    return len(ids)


def clear(source: Source) -> None:
    """Drop a source's collection entirely.

    Deleting by id needs the ids; dropping the collection does not, and
    get_or_create rebuilds it empty on the next call.
    """
    client().delete_collection(collection_name(source))
    logger.info(f"Dropped collection {collection_name(source)}")
