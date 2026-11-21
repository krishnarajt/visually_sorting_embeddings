"""Cosine similarity and vector arithmetic, always in the original dimensions.

This module is the counterweight to app/services/projection.py. Everything
there throws information away to make a picture; everything here works on the
full vectors, so the numbers the UI puts next to a word are the real ones.

That division is the single most important design decision in this app. If the
similarities were read off the 3-D coordinates, two words could look adjacent
and be reported as similar purely because a projection put them there — the
visualisation would be confirming itself. Instead the plot is a map you
navigate, and every similarity, neighbour list and analogy result is computed
from the vectors the model actually produced.
"""

import logging

import numpy as np

from app.services import store
from app.services.embedding import Source, glove, key_for

logger = logging.getLogger(__name__)


def unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    u = unit(np.asarray(vectors, dtype=np.float32))
    return np.clip(u @ u.T, -1.0, 1.0)


def top_pairs(
    words: list[str], matrix: np.ndarray, limit: int = 12
) -> list[dict]:
    """The most and least similar pairs on screen.

    A 40x40 similarity matrix is 780 numbers and no one reads it. The extremes
    are where the interesting cases live — the pair that surprises you and the
    pair that should be opposites and is not.
    """
    n = len(words)
    if n < 2:
        return []
    iu = np.triu_indices(n, k=1)
    scores = matrix[iu]
    order = np.argsort(-scores)
    picks = list(order[:limit]) + list(order[-limit:][::-1])
    seen: set[int] = set()
    out: list[dict] = []
    for idx in picks:
        if idx in seen:
            continue
        seen.add(int(idx))
        i, j = int(iu[0][idx]), int(iu[1][idx])
        out.append({"a": words[i], "b": words[j], "similarity": float(scores[idx])})
    return out


def _candidate_pool(
    source: Source, exclude: set[str]
) -> tuple[list[str], np.ndarray, str]:
    """The words a neighbour or analogy query is allowed to answer with.

    GloVe differs from every other source here in a way that matters to the
    user, so the answer carries a label saying which pool was searched:

      GloVe — the entire 50k-word vocabulary on disk. This is what makes
              king - man + woman satisfying: queen comes back because it is
              genuinely the nearest point in the space, not because it
              happened to be on screen.
      rest  — only what is in the index, because there is no local vocabulary
              and enumerating one would mean 50k words through an API or a
              transformer. So the answer is the nearest word *you have added*,
              which is a weaker claim and is presented as one.

    A pivot source inherits its base provider's pool, which is the right
    answer rather than a convenient one: a translate-to-English index over
    GloVe *is* an English GloVe index, so searching all 50k English words is
    exactly what querying it does. That is the pivot's whole selling point —
    the vocabulary stops being the model's problem.
    """
    if source.provider == "glove":
        return [], np.zeros((0, 0), dtype=np.float32), "the full 50k GloVe vocabulary"
    words, vectors = store.fetch_all(source)
    keep = [i for i, w in enumerate(words) if w not in exclude]
    return (
        [words[i] for i in keep],
        vectors[keep] if len(keep) else vectors[:0],
        f"the {len(keep)} words in this index",
    )


def nearest(
    source: Source, vector: np.ndarray, topk: int, exclude: set[str]
) -> tuple[list[dict], str]:
    """Nearest words to an arbitrary vector, with the pool that was searched."""
    if source.provider == "glove":
        hits = glove.nearest(vector, topk, exclude)
        return (
            [{"word": w, "similarity": s} for w, s in hits],
            "the full 50k GloVe vocabulary",
        )
    words, vectors, pool = _candidate_pool(source, exclude)
    if not words:
        return [], pool
    sims = unit(vectors.astype(np.float32)) @ unit(
        np.asarray(vector, dtype=np.float32)
    )
    order = np.argsort(-sims)[:topk]
    return (
        [{"word": words[i], "similarity": float(sims[i])} for i in order],
        pool,
    )


def neighbours_of(
    source: Source, word: str, topk: int
) -> tuple[list[dict], str]:
    key = key_for(source, word)
    if source.pivot:
        # In a pivot index the query is translated too, so the vector to search
        # from is the one filed under the original word — which holds the
        # embedding of its English translation. Reading it from the store
        # rather than recomputing it is what guarantees the query took the same
        # route through the translator as the documents did.
        found = store.fetch(source, [key])
        if key not in found:
            raise KeyError(f"{word!r} has not been embedded in this index yet")
        vector = found[key]
    elif source.provider == "glove":
        vector = glove.get(key)
        if vector is None:
            # Averaging a phrase is fine for plotting but meaningless as the
            # centre of a neighbour query, so say so instead of guessing.
            raise KeyError(f"{word!r} is not in the GloVe vocabulary")
    else:
        found = store.fetch(source, [key])
        if key not in found:
            raise KeyError(f"{word!r} has not been embedded in this index yet")
        vector = found[key]

    exclude = {key}
    if source.pivot:
        # Otherwise the top hit for "собака" is "dog", which is not a
        # neighbour — it is the same vector under the name it was filed as.
        from app.services import translate

        english = translate.known_english(word)
        if english:
            exclude.add(english)
            exclude.add(english.lower())
    return nearest(source, vector, topk, exclude)


def arithmetic(
    source: Source, terms: list[tuple[str, int]], topk: int
) -> dict:
    """Signed vector arithmetic: king - man + woman, and what it lands on.

    Returns the result vector's nearest words plus, for every input term, how
    similar the result is to it. That second list is the part that makes the
    failures legible: when an analogy "fails", the usual reason is that the
    result stayed closest to the term it started from, and seeing
    similarity(result, king) = 0.86 explains the answer better than the answer
    does.
    """
    vectors: list[np.ndarray] = []
    signs: list[int] = []
    resolved: list[str] = []
    unknown: list[str] = []

    for raw, sign in terms:
        key = key_for(source, raw)
        vector: np.ndarray | None = None
        if source.provider == "glove" and not source.pivot:
            vector = glove.get(key)
        else:
            # Pivot sources included: the vector filed under a term is the
            # embedding of its translation, and that is the one the arithmetic
            # has to use.
            vector = store.fetch(source, [key]).get(key)
        if vector is None:
            unknown.append(raw)
            continue
        vectors.append(np.asarray(vector, dtype=np.float32))
        signs.append(sign)
        resolved.append(key)

    if not vectors:
        return {
            "result": [],
            "unknown": unknown,
            "terms": [],
            "pool": "",
            "vector": None,
        }

    # Deliberately on the raw vectors, not normalised ones. This is the
    # arithmetic the word2vec and GloVe papers performed, and normalising each
    # term first is a different (also defensible) operation that gives slightly
    # different answers. Matching the original is the point of offering GloVe.
    combined = np.zeros_like(vectors[0])
    for vector, sign in zip(vectors, signs):
        combined = combined + sign * vector

    hits, pool = nearest(source, combined, topk, set(resolved))
    combined_u = unit(combined)
    return {
        "result": hits,
        "unknown": unknown,
        "terms": [
            {
                "word": word,
                "sign": sign,
                # How much of each input survives in the answer.
                "similarity_to_result": float(unit(vector) @ combined_u),
            }
            for word, sign, vector in zip(resolved, signs, vectors)
        ],
        "pool": pool,
        "norm": float(np.linalg.norm(combined)),
    }

def identical_groups(words: list[str], vectors: np.ndarray) -> list[list[str]]:
    """Groups of words whose vectors are byte-identical.

    Rare in a native index and the normal case in a pivot one: translating
    `eau`, `agua`, `Wasser`, `вода` and `पानी` all produce the literal string
    `water`, a model handed the same string returns the same vector, and six
    words become one point drawn six times.
    """
    if vectors.shape[0] < 2:
        return []
    _, inverse = np.unique(vectors, axis=0, return_inverse=True)
    buckets: dict[int, list[str]] = {}
    for word, group in zip(words, inverse.ravel().tolist()):
        buckets.setdefault(int(group), []).append(word)
    return [members for members in buckets.values() if len(members) > 1]


def describe_identical_groups(
    groups: list[list[str]],
    total_words: int,
    method_label: str,
    drawn_positions: int,
    translated: bool,
) -> str:
    """The note the UI shows over a view containing stacked points.

    Two halves, and the second is the one worth having. The first explains the
    clumps, which a user would otherwise have no way to tell from clumps the
    model chose to make. The second says what this particular projection did
    about them — and a method that spreads identical vectors apart is drawing a
    distinction that is not in the data, which it can do while reporting a high
    trustworthiness. On the translated translations grid UMAP gives all 48
    words their own position and scores 0.94 for it.

    The text lives here rather than in the route because the offline bundle
    needs the same sentence, and two copies of a sentence this load-bearing
    would eventually disagree.
    """
    if not groups:
        return ""
    example = groups[0]
    note = (
        f"{len(groups)} groups of words here share a vector exactly — "
        f"{', '.join(example[:4])}"
        + (f" and {len(example) - 4} more" if len(example) > 4 else "")
        + f" are one point drawn {len(example)} times"
    )
    if translated:
        note += (
            ", because translation turned them all into the same English "
            "string before the model saw them"
        )
    if drawn_positions >= total_words:
        note += (
            f". {method_label} has nonetheless given all {total_words} of them "
            "separate positions: every gap you can see inside a clump is this "
            "projection's invention, not the model's."
        )
    elif drawn_positions > len(groups):
        note += (
            f". {method_label} has drawn {drawn_positions} positions for "
            f"{len(groups)} distinct vectors, so some of the spread inside a "
            "clump is the projection's rather than the model's."
        )
    else:
        note += f". {method_label} draws them on top of each other, correctly."
    return note
