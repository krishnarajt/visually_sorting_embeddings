"""Flattening 300 to 3072 dimensions down to three, six different ways.

This module is the point of the application. Every method here takes the same
matrix and produces a different picture of it, and none of the pictures is the
truth — three numbers cannot hold what 1536 held. What differs is *which*
property each method sacrifices:

  pca     keeps the directions of greatest variance. Global shape survives,
          fine structure does not, and it reports honestly how much of the
          variance the three axes actually account for (usually 10-30%).
  tsne    keeps local neighbourhoods and abandons global geometry. Makes the
          prettiest clusters and the most misleading gaps: distance *between*
          two t-SNE clusters means nothing at all.
  umap    keeps local neighbourhoods and some global structure. Usually the
          best single answer, and the hardest to reason about.
  mds     directly minimises the disagreement between true cosine distance and
          drawn distance. The only method whose axes you may measure with a
          ruler, and it reports its residual as stress.
  random  three random directions. The control condition — this is what the
          data looks like with no method at all, and the comparison is why the
          others are worth their cost.
  raw     three dimensions of the original vector, picked by index. Shows that
          individual dimensions of a learned embedding mean nothing on their
          own.

Every projection also carries `trustworthiness` and `distance_correlation`,
computed against the full-dimensional vectors. Those two numbers are what stop
the visualisation from being a lie: they say how much of the real structure
survived this particular flattening.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.random_projection import GaussianRandomProjection

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    # One line, shown under the method picker.
    blurb: str
    # The longer "what am I looking at" text, shown in the explain panel.
    detail: str
    # Tunables the UI renders as sliders, with bounds.
    params: list[dict[str, Any]] = field(default_factory=list)
    # False when the import is missing at runtime (UMAP only).
    deterministic: bool = True
    # Minimum number of points before the method can run at all.
    min_points: int = 3


METHODS: list[Method] = [
    Method(
        id="pca",
        label="PCA",
        blurb="Linear. The three directions the words vary along most.",
        detail=(
            "Principal component analysis rotates the space so the first axis is "
            "the direction of greatest variance, then the second, then the third, "
            "and throws the rest away. It is linear, instant and completely "
            "deterministic — the same words always give the same picture. The cost "
            "shows up in the variance readout: three axes out of 1536 typically "
            "retain 10-30% of the variation, so most of what distinguishes these "
            "words has been discarded to fit them on a screen. Distances you see "
            "are real but heavily compressed, and unrelated words can land on top "
            "of each other because they happen to agree on three axes."
        ),
        min_points=3,
    ),
    Method(
        id="tsne",
        label="t-SNE",
        blurb="Neighbourhoods only. Beautiful clusters, meaningless gaps.",
        detail=(
            "t-SNE converts distances into probabilities of being neighbours, then "
            "arranges points in 3-D so those probabilities match as closely as it "
            "can. It is very good at what it promises and routinely misread. Two "
            "things are true and worth holding onto: points inside a clump really "
            "are near each other in the original space, and the distance between "
            "two clumps is an artefact of the layout. A large gap does not mean the "
            "groups are unrelated. Perplexity is roughly 'how many neighbours "
            "count' — raise it and clusters merge, lower it and they shatter."
        ),
        params=[
            {
                "id": "perplexity",
                "label": "Perplexity",
                "min": 2,
                "max": 50,
                "step": 1,
                "default": 15,
                "hint": "Roughly how many neighbours each point tries to stay near.",
            }
        ],
        deterministic=False,
        min_points=5,
    ),
    Method(
        id="umap",
        label="UMAP",
        blurb="Neighbourhoods plus some global shape. Usually the best picture.",
        detail=(
            "UMAP builds a graph of each point's nearest neighbours and then finds "
            "a 3-D layout of that graph. In practice it keeps t-SNE's clean local "
            "clusters while preserving more of the relationship between them, which "
            "makes it the one to reach for when you want a single view. It is also "
            "the hardest to argue about: the layout depends on the graph, and the "
            "graph depends on n_neighbors. Low values give many small islands, high "
            "values pull everything into one continent."
        ),
        params=[
            {
                "id": "n_neighbors",
                "label": "Neighbours",
                "min": 2,
                "max": 60,
                "step": 1,
                "default": 15,
                "hint": "Small values favour local detail, large values global shape.",
            },
            {
                "id": "min_dist",
                "label": "Min distance",
                "min": 0.0,
                "max": 0.99,
                "step": 0.01,
                "default": 0.1,
                "hint": "How tightly points may be packed together.",
            },
        ],
        deterministic=False,
        min_points=5,
    ),
    Method(
        id="mds",
        label="MDS",
        blurb="Matches real cosine distances as closely as three axes allow.",
        detail=(
            "Multidimensional scaling takes the full table of true cosine distances "
            "and looks for the 3-D arrangement whose straight-line distances match "
            "it best. That makes it the only method on this list whose picture you "
            "are allowed to measure: if two words look twice as far apart as two "
            "others, they roughly are. The honesty has a price — it reports the "
            "residual disagreement as stress, and on real embedding data the stress "
            "is high, because the geometry genuinely does not fit in three "
            "dimensions. Below about 0.1 is a good fit, above 0.2 means the "
            "distances on screen are only suggestive."
        ),
        min_points=3,
    ),
    Method(
        id="random",
        label="Random projection",
        blurb="Three random directions. The control condition.",
        detail=(
            "Multiply the vectors by a random matrix and keep three of the results. "
            "This is not a straw man: the Johnson-Lindenstrauss lemma says random "
            "projection preserves distances remarkably well — into a few hundred "
            "dimensions. Into three it collapses into a blob, which is the useful "
            "part. Flip between this and UMAP on the same words: everything you can "
            "see in one and not the other is what the method contributed, rather "
            "than something that was lying around in the data."
        ),
        min_points=3,
    ),
    Method(
        id="raw",
        label="Raw dimensions",
        blurb="Three dimensions of the actual vector, chosen by index or by variance.",
        detail=(
            "No reduction at all — just three dimensions of the raw vector as x, y "
            "and z. The result is noise, and that is the lesson. Unlike the columns "
            "of a spreadsheet, a single dimension of a learned embedding has no "
            "meaning of its own; the information lives in the combination of all of "
            "them. "
            "Switching the picker to the most varied dimensions is the fair version "
            "of the test — instead of arbitrary indices it hands raw mode the three "
            "best dimensions available for these particular words, and the readout "
            "prints what they explain against what three PCA components explain. On "
            "LaBSE with the translations set that is 1.1% against 46.1%, so the best "
            "three dimensions in the model lose to three learned combinations by a "
            "factor of forty. That ratio is the whole reason the other five methods "
            "exist, and it is worth having as a number rather than an assertion."
        ),
        params=[
            {
                "id": "select",
                "label": "Pick by",
                "type": "choice",
                "default": "index",
                "options": [
                    {"value": "index", "label": "index"},
                    {"value": "variance", "label": "most varied"},
                ],
                "hint": (
                    "index uses the sliders; most varied uses the three dimensions "
                    "that differ most across the words on screen"
                ),
            },
            {"id": "dim_x", "label": "X = dim", "min": 0, "max": 299, "step": 1, "default": 0},
            {"id": "dim_y", "label": "Y = dim", "min": 0, "max": 299, "step": 1, "default": 1},
            {"id": "dim_z", "label": "Z = dim", "min": 0, "max": 299, "step": 1, "default": 2},
        ],
        min_points=1,
    ),
]

METHODS_BY_ID: dict[str, Method] = {m.id: m for m in METHODS}
DEFAULT_METHOD_ID = "umap"


class ProjectionError(ValueError):
    """Bad request for a projection, with a message meant for the UI."""


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
    except Exception:  # pragma: no cover - depends on the install
        return False
    return True


# ---------------------------------------------------------------------------
# Quality metrics
#
# These are computed on the full-dimensional vectors against the 3-D output,
# and they are the reason this app can claim to show similarity rather than
# just draw dots.
# ---------------------------------------------------------------------------


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    # Floating point can leave tiny negatives, which MDS rejects outright.
    return np.maximum(distance, 0.0)


def _quality(vectors: np.ndarray, coords: np.ndarray) -> dict[str, Any]:
    n = vectors.shape[0]
    out: dict[str, Any] = {}

    # trustworthiness asks: of the k nearest neighbours a point has on screen,
    # how many were genuinely among its k nearest in the original space? 1.0
    # is perfect, and roughly 1/k is what random noise scores.
    if n >= 6:
        k = max(2, min(10, (n - 1) // 2))
        try:
            out["trustworthiness"] = float(
                trustworthiness(vectors, coords, n_neighbors=k, metric="cosine")
            )
            out["trustworthiness_k"] = k
        except Exception as exc:  # pragma: no cover - guards sklearn edge cases
            logger.warning(f"trustworthiness failed: {exc}")

    # And the global counterpart: rank correlation between every true cosine
    # distance and the drawn distance. High trustworthiness with low distance
    # correlation is the exact signature of t-SNE — right neighbours, wrong
    # distances — and being able to see that trade-off as two numbers is worth
    # more than either alone.
    if n >= 4:
        true_d = squareform(_cosine_distance_matrix(vectors), checks=False)
        drawn = coords[:, None, :] - coords[None, :, :]
        drawn_d = squareform(np.sqrt((drawn**2).sum(-1)), checks=False)
        if np.std(true_d) > 1e-12 and np.std(drawn_d) > 1e-12:
            rho = spearmanr(true_d, drawn_d).statistic
            if rho is not None and not np.isnan(rho):
                out["distance_correlation"] = float(rho)
    return out


# ---------------------------------------------------------------------------
# Framing and stability
# ---------------------------------------------------------------------------


def _frame(coords: np.ndarray) -> np.ndarray:
    """Centre on the origin and scale so the furthest point sits at radius 1.

    Methods disagree wildly about scale — PCA returns whatever the data's units
    were, t-SNE returns tens, UMAP single digits. Normalising here means the
    camera does not have to be re-framed every time the method changes, which
    is the difference between comparing two pictures and hunting for the second
    one. Shape is untouched: this is a uniform scale, not a per-axis one, so it
    cannot make a method look better than it is.
    """
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords - coords.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(coords, axis=1))) if len(coords) else 0.0
    if radius > 1e-12:
        coords = coords / radius
    return coords


def _align(coords: np.ndarray, words: list[str], anchor: dict[str, list[float]]) -> np.ndarray:
    """Rotate/reflect a layout to best match where the words were last time.

    t-SNE and UMAP have no canonical orientation, so adding a single word can
    return a layout that is correct and looks completely different — the whole
    cloud mirrored or spun. Users read that as the meaning having changed. An
    orthogonal Procrustes fit against the previous frame removes the part of
    the change that is arbitrary while leaving the part that is real, because
    rotation and reflection are exactly the transformations that carry no
    information about the data.
    """
    shared = [i for i, w in enumerate(words) if w in anchor]
    if len(shared) < 3:
        return coords
    from scipy.linalg import orthogonal_procrustes

    current = coords[shared]
    previous = np.asarray([anchor[words[i]] for i in shared], dtype=np.float64)
    current_c = current - current.mean(axis=0)
    previous_c = previous - previous.mean(axis=0)
    if np.linalg.norm(current_c) < 1e-12 or np.linalg.norm(previous_c) < 1e-12:
        return coords
    rotation, _ = orthogonal_procrustes(current_c, previous_c)
    return (coords - current.mean(axis=0)) @ rotation


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------


def _project_pca(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    n_components = min(3, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=settings.random_seed)
    coords = pca.fit_transform(vectors)
    coords = _pad_to_3(coords)
    ratios = [float(r) for r in pca.explained_variance_ratio_]
    return coords, {
        "explained_variance_ratio": ratios,
        "explained_variance_total": float(sum(ratios)),
    }


def _project_tsne(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    n = vectors.shape[0]
    requested = float(params.get("perplexity", 15))
    # sklearn requires perplexity < n_samples and is unstable as it approaches
    # it. A third of the sample is the usual rule of thumb.
    perplexity = max(2.0, min(requested, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=settings.random_seed,
        max_iter=1000,
    )
    coords = tsne.fit_transform(vectors)
    meta = {
        "perplexity": perplexity,
        "kl_divergence": float(tsne.kl_divergence_),
        "iterations": int(tsne.n_iter_),
    }
    if abs(perplexity - requested) > 1e-9:
        meta["clamped"] = (
            f"Perplexity clamped from {requested:g} to {perplexity:g} — it has to "
            f"stay well under the {n} points on screen."
        )
    return coords, meta


def _project_umap(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    try:
        import umap
    except Exception as exc:
        raise ProjectionError(
            "UMAP is not installed in this environment. `pip install umap-learn`, "
            "or use one of the other five methods."
        ) from exc
    n = vectors.shape[0]
    requested = int(params.get("n_neighbors", 15))
    n_neighbors = max(2, min(requested, n - 1))
    min_dist = float(params.get("min_dist", 0.1))
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=settings.random_seed,
        init="spectral",
    )
    coords = reducer.fit_transform(vectors)
    meta = {"n_neighbors": n_neighbors, "min_dist": min_dist}
    if n_neighbors != requested:
        meta["clamped"] = (
            f"Neighbours clamped from {requested} to {n_neighbors} — cannot exceed "
            f"the {n} points on screen."
        )
    return np.asarray(coords, dtype=np.float64), meta


def _project_mds(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    distances = _cosine_distance_matrix(vectors)
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=settings.random_seed,
        n_init=4,
        max_iter=300,
        normalized_stress=False,
    )
    coords = mds.fit_transform(distances)
    # sklearn refuses to normalise stress for metric MDS, and raw stress is a
    # sum of squares whose size depends on how many words are on screen — 0.4
    # would mean nothing on its own. Kruskal's stress-1 is that residual over
    # the total sum of squared distances, which is the scale the familiar
    # "under 0.1 is a good fit" rule of thumb is stated on.
    total = float((squareform(distances, checks=False) ** 2).sum())
    stress1 = float(np.sqrt(mds.stress_ / total)) if total > 1e-12 else 0.0
    return coords, {"stress": stress1, "stress_scale": "kruskal-1"}


def _project_random(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    projector = GaussianRandomProjection(
        n_components=3, random_state=settings.random_seed
    )
    coords = projector.fit_transform(vectors)
    return coords, {
        "note": "Three random directions, seeded so the blob at least holds still."
    }


def _project_raw(vectors: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Three literal dimensions of the vector, and how little they hold.

    Two ways to choose them. `index` is the sliders, and the point of it is
    that the picture never becomes interpretable however far you drag. But an
    arbitrary index is a soft version of the test — someone is entitled to
    think a better choice of dimension would have worked. `variance` makes the
    strongest case the method can: the three dimensions that differ most across
    exactly these words, which is the best any axis-aligned triple can do here.

    Either way the meta carries `explained` against `pca_explained` on the same
    words, because "the result is noise" is an assertion and "1.1% against
    46.1%" is a measurement.
    """
    n, d = vectors.shape
    mode = str(params.get("select", "index"))

    centred = vectors - vectors.mean(axis=0)
    variance = centred.var(axis=0)
    total = float(variance.sum())

    if mode == "variance":
        # Sorted by index after selection so the axes stay in a stable order
        # when the word list changes and the ranking shuffles.
        dims = sorted(int(i) for i in np.argsort(-variance)[:3])
    else:
        dims = [
            int(params.get("dim_x", 0)) % d,
            int(params.get("dim_y", 1)) % d,
            int(params.get("dim_z", 2)) % d,
        ]

    meta: dict[str, Any] = {
        "dims": dims,
        "of": d,
        "select": mode,
        # Deduplicated: three sliders on the same value is one axis, and
        # reporting it as three would overstate what is on screen.
        "explained": round(float(variance[sorted(set(dims))].sum() / total), 5)
        if total
        else 0.0,
    }

    # The comparison that makes the lesson land, on the same words. PCA needs
    # at least as many samples as components, so a two-word scene simply does
    # not get the number rather than getting a misleading one.
    components = min(3, n - 1, d)
    if components >= 1 and total:
        baseline = PCA(n_components=components, random_state=settings.random_seed)
        baseline.fit(centred)
        pca_explained = float(baseline.explained_variance_ratio_.sum())
        meta["pca_explained"] = round(pca_explained, 5)
        if meta["explained"] > 0:
            meta["pca_better_by"] = round(pca_explained / meta["explained"], 1)

    return vectors[:, dims].astype(np.float64), meta


_PROJECTORS = {
    "pca": _project_pca,
    "tsne": _project_tsne,
    "umap": _project_umap,
    "mds": _project_mds,
    "random": _project_random,
    "raw": _project_raw,
}


def _pad_to_3(coords: np.ndarray) -> np.ndarray:
    """Zero-fill missing axes.

    Two words cannot have three principal components. Rather than refuse, drop
    them on a line through the origin — which is the honest picture of a
    two-point space anyway.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] >= 3:
        return coords[:, :3]
    padding = np.zeros((coords.shape[0], 3 - coords.shape[1]))
    return np.hstack([coords, padding])


# ---------------------------------------------------------------------------
# Clustering, for colour
# ---------------------------------------------------------------------------


def cluster(vectors: np.ndarray, k: int) -> list[int]:
    """k-means labels, computed on the full vectors rather than the 3-D ones.

    Colouring by clusters found in the original space is what turns the plot
    into evidence: when same-coloured points land together, that agreement is
    between two independent readings of the data. Clustering the 3-D
    coordinates instead would only ever confirm the projection to itself.
    """
    n = vectors.shape[0]
    k = max(1, min(int(k), n))
    if k <= 1 or n < 2:
        return [0] * n
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # k-means minimises euclidean distance; on unit-normalised vectors that is
    # a monotone function of cosine distance, which is the metric everything
    # else in this app uses.
    km = KMeans(n_clusters=k, random_state=settings.random_seed, n_init=10)
    return [int(label) for label in km.fit_predict(vectors / norms)]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: OrderedDict[str, tuple[np.ndarray, dict]] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(source_id: str, method_id: str, words: list[str], params: dict) -> str:
    payload = "|".join(
        [
            source_id,
            method_id,
            ",".join(words),
            ",".join(f"{k}={params[k]}" for k in sorted(params)),
            str(settings.random_seed),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def project(
    source_id: str,
    method_id: str,
    words: list[str],
    vectors: np.ndarray,
    params: dict | None = None,
    anchor: dict[str, list[float]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Project `vectors` to 3-D, with quality metrics and a stable orientation."""
    params = params or {}
    method = METHODS_BY_ID.get(method_id)
    if method is None:
        raise ProjectionError(f"No such method {method_id!r}")
    n = vectors.shape[0]
    if n == 0:
        raise ProjectionError("Nothing to project — add some words first.")
    if n < method.min_points:
        raise ProjectionError(
            f"{method.label} needs at least {method.min_points} words to say "
            f"anything; there {'is' if n == 1 else 'are'} {n}. PCA and MDS work "
            "on fewer."
        )
    if n > settings.max_words_per_projection:
        raise ProjectionError(
            f"{n} words is past the {settings.max_words_per_projection} limit for "
            "one projection."
        )

    key = _cache_key(source_id, method_id, words, params)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            coords, meta = hit
            coords = coords.copy()
            meta = {**meta, "cached": True}
    if hit is None:
        coords, meta = _PROJECTORS[method.id](vectors, params)
        coords = _frame(_pad_to_3(coords))
        meta = {**meta, **_quality(vectors, coords), "cached": False}
        with _cache_lock:
            _cache[key] = (coords.copy(), {k: v for k, v in meta.items() if k != "cached"})
            while len(_cache) > settings.projection_cache_size:
                _cache.popitem(last=False)

    if anchor:
        coords = _frame(_align(coords, words, anchor))
    return coords, meta


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()
