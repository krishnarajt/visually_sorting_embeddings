"""Which dimensions a word lives on, and who lives there with it.

This exists because of a reasonable intuition that the `raw` projection method
only half-refutes. Raw mode shows that *picking* three dimensions to draw is
hopeless — on LaBSE the three most varied of 768 explain 1.1% of the variance
where three PCA components explain 46.1%. It is easy to read that as "the
dimensions mean nothing", and that is too strong.

They mean nothing *individually*. But the **set** of dimensions a word drives
hardest is partly shared by words that mean the same thing, and the overlap
carries real information about meaning. Measured on LaBSE over the
48-word translations grid, top-5 overlap against cosine:

    water / вода    cosine 0.98   shares 5 of 5
    water / eau     cosine 0.97   shares 4
    book  / किताब    cosine 0.94   shares 4
    book  / книга   cosine 0.95   shares 3
    book  / Buch    cosine 0.85   shares 2
    book  / livre   cosine 0.65   shares 1

Across all 48 words the rank correlation between overlap and cosine has a
median of **+0.53** on LaBSE (mean +0.50, range −0.14 to +0.83). It is positive
for every model here, and how positive tracks the same axis everything else in
this app does — median over the translations grid, top-5 overlap:

    minilm-multilingual   +0.69      labse                         +0.53
    glove-300             +0.51      cohere-embed-v3-multilingual  +0.42
    e5-multilingual-small +0.38      openai-3-large                +0.33
    bge-m3                +0.29      openai-3-small                +0.21

The two alignment-trained models come top, which is the same ordering their
silhouettes give in app/services/alignment.py, arrived at by a completely
different route. That is worth something. What it is not is a similarity
measure: a median of +0.5 with a per-word range from −0.14 to +0.83 means
individual words mislead badly, and cosine already answers "how similar" using
all 768 dimensions instead of five. What this adds is *which* dimensions, and
shared with whom, which cosine collapses away.

So the honest picture has two halves, and both are measured: meaning is spread
across a combination of dimensions rather than stored in any one of them
(`raw`), and the combination is still partly legible (here).

## Centring, and the hub dimensions

Ranking a word's dimensions by raw magnitude has an obvious objection.
Dimension 216 is in the top five of `water`, `dog` *and* `book` on LaBSE,
because it is simply large for almost every vector — these models carry a
handful of such "hub" or "rogue" dimensions holding a global offset rather than
anything about the word. An overlap that was only hubs would be an artefact.

Ranking instead by how far the word sits from the set mean, in units of that
dimension's spread, removes them for free: a dimension large for everyone
deviates for no one. That is an ordinary z-score, and it was the first thing
tried here.

It removes the signal too, which is the part worth knowing. When the comparison
set is a translation grid, the dimensions a concept's members share sit near the
set mean *by construction*, so what survives centring is mostly each word's own
idiosyncrasy — the median correlation falls from +0.53 to +0.40. Centring is
the right answer to the hub problem and the wrong answer to this question.

The objection turns out to be answerable directly instead, by recomputing the
overlap with the hub dimensions simply deleted. Doing that costs some of the
agreement and never all of it:

    glove-300    +0.51 -> +0.37        minilm-multilingual  +0.69 -> +0.54
    labse        +0.53 -> +0.53        bge-m3               +0.29 -> +0.28
    openai-3-small +0.21 -> +0.27  (the hubs were *hurting* this one)

So hubs inflate the counts for some models and carry none of the ordering for
others, and on one they are actively noise. There is no single answer, which is
why `shared_nonhub` is returned per pair rather than folded into a verdict —
and why the number of hub dimensions is worth seeing on its own. bge-m3 has
five of them in this test where e5-multilingual-small has none, a difference
between the two models that nothing else in this app surfaces.

A warning about reading any single word's figure: they scatter widely. GloVe's
overlap for `water` alone falls from +0.56 to +0.06 once its hub is removed,
which looks like a collapse and is one word out of forty-eight. The medians
above are the claim; the per-word number in the panel is an observation.

Everything here is computed on the full vectors, like app/services/similarity.py
and app/services/alignment.py, and never on a projection's coordinates.
"""

import logging

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _hub_dimensions(matrix: np.ndarray, top_k: int) -> list[int]:
    """Dimensions that land in most words' top-k by raw magnitude.

    Reported so the UI can mark them. A dimension that is "important" for
    ninety per cent of the vocabulary is not telling you anything about any
    particular word.
    """
    n = matrix.shape[0]
    if n == 0:
        return []
    counts = np.zeros(matrix.shape[1], dtype=np.int32)
    for row in np.abs(matrix):
        counts[np.argsort(-row)[:top_k]] += 1
    # Half the words is a deliberately loose bar: with eight words a dimension
    # shared by five is already saying more about the model than the word.
    return sorted(int(i) for i in np.where(counts >= max(2, n // 2))[0])


def profile(
    words: list[str],
    matrix: np.ndarray,
    word: str,
    top_k: int = 8,
) -> dict:
    """The dimension profile of one word against the others on screen.

    `matrix` is row-aligned with `words` and in the model's own dimensionality.
    The comparison set is whatever the caller passed, which means the z-scores
    are relative to the words in front of the user rather than to the whole
    vocabulary — the right frame for "what makes *this* word different from
    *these* words", and the wrong one for any claim about the model at large.
    That distinction is in the response as `relative_to`.
    """
    if word not in words:
        raise KeyError(f"{word!r} is not in this word list")
    if len(words) != matrix.shape[0]:
        raise ValueError(f"{len(words)} words but {matrix.shape[0]} vectors")

    index = words.index(word)
    n, d = matrix.shape
    top_k = max(1, min(top_k, d))

    vectors = np.asarray(matrix, dtype=np.float64)
    vector = vectors[index]

    mean = vectors.mean(axis=0)
    # A single word has no spread to measure against, and a dimension that is
    # constant across the set has none either; both fall back to 1.0 so the
    # z-score degrades to a plain centred value instead of dividing by zero.
    spread = vectors.std(axis=0) if n > 1 else np.ones(d)
    spread = np.where(spread < 1e-9, 1.0, spread)
    z = (vectors - mean) / spread

    by_magnitude = [int(i) for i in np.argsort(-np.abs(vector))[:top_k]]
    by_deviation = [int(i) for i in np.argsort(-np.abs(z[index]))[:top_k]]
    hubs = set(_hub_dimensions(vectors, top_k))

    unit = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    cosines = np.clip(unit @ unit[index], -1.0, 1.0)

    # Overlap is reported on *both* rankings, because they answer different
    # questions and only one of them is the question this module is named for.
    #
    #   by magnitude  — "do these two words drive the same dimensions hard?"
    #                   This is the one that tracks cosine, and it is the
    #                   interesting result: water/вода share 5 of 5, while
    #                   book/livre at cosine 0.65 share 1.
    #   by deviation  — "are these two words unusual on the same dimensions,
    #                   relative to the words on screen?" Much weaker, and for
    #                   a good reason: when the comparison set is mostly
    #                   translations of each other, their shared dimensions are
    #                   near the mean *by construction*, so what is left after
    #                   centring is mostly each word's own idiosyncrasy.
    #
    # Centring was originally applied to both, which removed the hub problem
    # and the signal along with it. Keeping both, with the hub dimensions
    # flagged rather than subtracted, is what makes the difference legible
    # instead of hidden.
    mine_magnitude = set(by_magnitude)
    mine_deviation = set(by_deviation)
    mine_nonhub = mine_magnitude - hubs

    shared: list[dict] = []
    for other in range(n):
        if other == index:
            continue
        theirs_magnitude = set(
            int(i) for i in np.argsort(-np.abs(vectors[other]))[:top_k]
        )
        theirs_deviation = set(
            int(i) for i in np.argsort(-np.abs(z[other]))[:top_k]
        )
        common = sorted(mine_magnitude & theirs_magnitude)
        shared.append({
            "word": words[other],
            "similarity": round(float(cosines[other]), 4),
            "shared": len(common),
            "dimensions": common,
            # The same count with the hub dimensions taken out, which is the
            # answer to "is the overlap just the dimensions everything shares".
            # It is not: on LaBSE the ordering survives their removal.
            "shared_nonhub": len(mine_nonhub & (theirs_magnitude - hubs)),
            "shared_deviation": len(mine_deviation & theirs_deviation),
        })
    shared.sort(key=lambda row: (-row["shared"], -row["similarity"]))

    # Does overlap actually track meaning, or is it noise that happens to look
    # ordered? Spearman over the pairs answers it, for each ranking, and needs
    # three pairs with some variation in both columns to mean anything.
    def _rho(key: str) -> float | None:
        if len(shared) < 3:
            return None
        xs = [row[key] for row in shared]
        ys = [row["similarity"] for row in shared]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            return None
        value = spearmanr(xs, ys).statistic
        return round(float(value), 4) if np.isfinite(value) else None

    agreement = {
        "by_magnitude": _rho("shared"),
        "by_magnitude_nonhub": _rho("shared_nonhub"),
        "by_deviation": _rho("shared_deviation"),
    }

    return {
        "word": word,
        "dims": d,
        "top_k": top_k,
        "relative_to": len(words),
        # The two rankings, so the hub problem is shown rather than described.
        "by_magnitude": [
            {
                "dimension": i,
                "value": round(float(vector[i]), 5),
                "hub": i in hubs,
            }
            for i in by_magnitude
        ],
        "by_deviation": [
            {
                "dimension": i,
                "value": round(float(vector[i]), 5),
                "z": round(float(z[index][i]), 3),
                "hub": i in hubs,
            }
            for i in by_deviation
        ],
        # How much of the ranking-by-magnitude answer was hub dimensions that
        # say nothing about this word in particular.
        "hub_dimensions": sorted(hubs & set(by_magnitude)),
        "shared": shared,
        # Rank correlation between dimension overlap and cosine, per ranking.
        # The headline is `by_magnitude`; `by_deviation` is kept beside it
        # because a reader who assumes centring must be the better choice
        # deserves to see what it costs.
        "agreement": agreement,
    }
