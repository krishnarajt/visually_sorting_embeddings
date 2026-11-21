"""Does this model sort by meaning or by language? Measured, not eyeballed.

Every number here is computed from the full-dimensional vectors, never from
the three coordinates a projection produced — the same rule that governs
app/services/similarity.py, and for the same reason. A plot can put all the
Russian in one corner because UMAP felt like it. Whether the *vectors* put all
the Russian in one corner is a different question, and it is the one worth
answering.

## The four headline numbers, and why it takes four

A word in a `Lexicon` carries two labels: its concept and its language. So
every pair of words is one of three kinds — same concept (translations), same
language (neighbours in a dictionary), or unrelated. A model sorts by meaning
if the first kind is closer than the second.

  translation_p_at_1      Of the words in other languages, is the nearest one
                          this word's own translation? The retrieval question,
                          and the one that decides whether a search index
                          built on this model works at all.

  gap                     Mean same-concept cosine minus mean same-language
                          cosine. Answers a different question: by *how much*
                          does meaning win? A model can score 100% on
                          retrieval with a gap near zero —
                          multilingual-e5-small does exactly that, putting
                          every pair of strings above 0.8.
                          Careful with the conclusion, though: a small gap
                          does *not* mean no usable threshold exists. It means
                          the two distributions are close *in their means*.
                          e5's best single cutoff sits at 0.895 and reaches
                          F1 0.861 on the translations grid, comfortably ahead
                          of text-embedding-3-large's 0.762 — so the gap
                          understates e5 because its distributions are narrow
                          and still separable. Read it as "where is the
                          boundary and how obvious is it", not as "is there
                          one".

  meaning_beats_language  The scale-free version of `gap`. For each word, take
                          every same-concept similarity against every
                          same-language similarity and count how often meaning
                          wins. 0.5 is a coin flip, 1.0 is total separation.
                          Immune to a model that compresses all its cosines
                          into a narrow band, which is why it is reported
                          next to `gap` rather than instead of it.

  silhouette_by_*         Cluster the words by concept, then by language, and
                          ask which labelling the geometry actually agrees
                          with. The one number that can come out *negative*,
                          which is the interesting case: it means the clusters
                          are real and you have named them wrongly. GloVe
                          scores negative by concept and positive by language.

Reporting one of these would be easy and would mislead. Retrieval accuracy
alone flatters e5; the gap alone flatters nothing and punishes any model with
a compressed cosine range; the silhouettes alone say which labelling is
better without saying whether either is usable.

## Asymmetry

`by_language` reports each language against English rather than against all
the others, because that is the question a pivot index makes urgent: English
is where the translation lands, so how far each language has to travel to get
there is the thing that differs between the two designs.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entry:
    """One word in the grid, as it was actually stored and embedded."""

    text: str
    language: str
    concept: str


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _silhouette(distances: np.ndarray, labels: list[str]) -> float | None:
    """Silhouette on a precomputed cosine distance matrix, or None if undefined.

    sklearn needs at least two labels and at least one label with more than
    one member; a grid with one language or one concept in it has nothing to
    measure and says so rather than returning a zero that would read as
    "no structure".
    """
    unique = set(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    try:
        return float(silhouette_score(distances, labels, metric="precomputed"))
    except ValueError as exc:
        logger.debug(f"silhouette undefined: {exc}")
        return None


def evaluate(entries: list[Entry], vectors: np.ndarray) -> dict:
    """The full report for one model over one concept-by-language grid.

    `vectors` must be row-aligned with `entries` and in the model's original
    dimensionality.
    """
    n = len(entries)
    if n != vectors.shape[0]:
        raise ValueError(f"{n} entries but {vectors.shape[0]} vectors")

    concepts = np.array([e.concept for e in entries])
    languages = np.array([e.language for e in entries])
    texts = [e.text for e in entries]

    if n < 4 or len(set(concepts)) < 2 or len(set(languages)) < 2:
        return {
            "measurable": False,
            "reason": (
                f"{n} words across {len(set(concepts))} concepts and "
                f"{len(set(languages))} languages is not enough to separate "
                "the two. This model could not embed most of the word set — "
                "which is itself the answer for it."
            ),
            "words": n,
            "concepts": len(set(concepts)),
            "languages": len(set(languages)),
        }

    unit = _unit(np.asarray(vectors, dtype=np.float32))
    sims = np.clip(unit @ unit.T, -1.0, 1.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    same_concept = (concepts[:, None] == concepts[None, :]) & off_diagonal
    same_language = (languages[:, None] == languages[None, :]) & off_diagonal
    # A same-concept pair is by construction cross-language and a
    # same-language pair cross-concept, since a grid has one word per cell.
    # Intersecting them anyway keeps that an assertion rather than a belief.
    translation_pairs = same_concept & ~same_language
    dictionary_pairs = same_language & ~same_concept

    # --- retrieval -------------------------------------------------------
    hits: list[str] = []
    confusions: list[dict] = []
    asked = 0
    for i in range(n):
        candidates = np.where(languages != languages[i])[0]
        if not len(candidates):
            continue
        asked += 1
        best = candidates[int(np.argmax(sims[i, candidates]))]
        if concepts[best] == concepts[i]:
            hits.append(texts[i])
        else:
            # What it retrieved instead, and what it should have. A list of
            # the model's actual mistakes beats a percentage for working out
            # why it made them — they are almost never random.
            wanted = [
                j
                for j in candidates
                if concepts[j] == concepts[i]
            ]
            confusions.append({
                "word": texts[i],
                "language": languages[i],
                "concept": concepts[i],
                "retrieved": texts[best],
                "retrieved_language": str(languages[best]),
                "retrieved_concept": str(concepts[best]),
                "similarity": round(float(sims[i, best]), 4),
                "best_translation": texts[max(wanted, key=lambda j: sims[i, j])]
                if wanted
                else "",
                "best_translation_similarity": round(
                    float(max(sims[i, j] for j in wanted)), 4
                )
                if wanted
                else 0.0,
            })

    # --- separation ------------------------------------------------------
    meaning = float(sims[translation_pairs].mean()) if translation_pairs.any() else float("nan")
    language = float(sims[dictionary_pairs].mean()) if dictionary_pairs.any() else float("nan")

    wins = 0.0
    comparisons = 0
    for i in range(n):
        theirs = sims[i][translation_pairs[i]]
        others = sims[i][dictionary_pairs[i]]
        if not theirs.size or not others.size:
            continue
        # Every translation against every same-language word, ties at half.
        greater = (theirs[:, None] > others[None, :]).sum()
        equal = (theirs[:, None] == others[None, :]).sum()
        wins += float(greater) + 0.5 * float(equal)
        comparisons += theirs.size * others.size

    # Two different strings with the same vector. In a native index this is
    # vanishingly rare; in a pivot index it is the normal case, because "eau"
    # and "agua" and "Wasser" all translate to the literal string "water" and
    # a model given the same string twice returns the same vector twice. That
    # is not retrieval succeeding — it is the distinction being deleted before
    # retrieval was asked, and every score above goes to 1.0 as a result. It
    # has to be reported next to them or they are a lie of omission.
    identical = (sims >= 0.9999) & off_diagonal
    collapsed = [
        str(concept)
        for concept in dict.fromkeys(concepts)
        if _is_collapsed(concepts, identical, concept)
    ]

    distances = (1.0 - sims).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    # Cosine distance can come out microscopically negative from floating
    # point; sklearn rejects a precomputed matrix that is not non-negative.
    distances = np.maximum(distances, 0.0)

    return {
        "measurable": True,
        "words": n,
        "concepts": len(set(concepts)),
        "languages": len(set(languages)),
        "translation_p_at_1": round(len(hits) / asked, 4) if asked else 0.0,
        "translation_hits": len(hits),
        "translation_asked": asked,
        "mean_same_concept": round(meaning, 4),
        "mean_same_language": round(language, 4),
        "gap": round(meaning - language, 4),
        "meaning_beats_language": (
            round(wins / comparisons, 4) if comparisons else None
        ),
        "identical_pairs": int(identical.sum() // 2),
        "collapsed_concepts": collapsed,
        "distinct_vectors": int(n - identical.sum() // 2) if collapsed else n,
        "silhouette_by_concept": _round(_silhouette(distances, list(concepts))),
        "silhouette_by_language": _round(_silhouette(distances, list(languages))),
        "by_language": _by_language(entries, sims, concepts, languages, texts),
        "by_concept": _by_concept(sims, concepts, translation_pairs),
        "confusions": confusions,
    }


def _is_collapsed(
    concepts: np.ndarray, identical: np.ndarray, concept: str
) -> bool:
    """Whether every member of a concept shares one vector with every other."""
    rows = np.where(concepts == concept)[0]
    if len(rows) < 2:
        return False
    block = identical[np.ix_(rows, rows)]
    return bool(block.sum() == len(rows) * (len(rows) - 1))


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _by_language(
    entries: list[Entry],
    sims: np.ndarray,
    concepts: np.ndarray,
    languages: np.ndarray,
    texts: list[str],
) -> list[dict]:
    """Each language measured against English: how far it has to travel.

    English gets a row too, with `to_english` left at None — it is the
    reference, and a 1.0 there would look like a score.
    """
    out: list[dict] = []
    english_by_concept = {
        concepts[i]: i for i in range(len(texts)) if languages[i] == "en"
    }
    for code in dict.fromkeys(languages):
        members = np.where(languages == code)[0]
        if code == "en":
            out.append({
                "language": "en",
                "words": int(len(members)),
                "to_english": None,
                "p_at_1": None,
            })
            continue
        scores = [
            float(sims[i, english_by_concept[concepts[i]]])
            for i in members
            if concepts[i] in english_by_concept
        ]
        # Retrieval restricted to English only: of the English words, is this
        # word's own translation the nearest? This is the pivot index's
        # question, asked of the native index.
        english_rows = sorted(english_by_concept.values())
        hits = 0
        for i in members:
            if not english_rows:
                continue
            best = max(english_rows, key=lambda j: sims[i, j])
            hits += int(concepts[best] == concepts[i])
        out.append({
            "language": str(code),
            "words": int(len(members)),
            "to_english": round(sum(scores) / len(scores), 4) if scores else None,
            "p_at_1": round(hits / len(members), 4) if len(members) else None,
        })
    return out


def _by_concept(
    sims: np.ndarray, concepts: np.ndarray, translation_pairs: np.ndarray
) -> list[dict]:
    """How tightly each concept holds together across languages.

    Sorted loosest first, because the concept a model struggles to recognise
    across languages is more informative than the one it finds easy — and it
    is usually a word whose translations are not quite synonyms.
    """
    out: list[dict] = []
    for concept in dict.fromkeys(concepts):
        rows = np.where(concepts == concept)[0]
        mask = np.zeros_like(translation_pairs)
        mask[np.ix_(rows, rows)] = True
        mask &= translation_pairs
        if not mask.any():
            continue
        out.append({
            "concept": str(concept),
            "words": int(len(rows)),
            "cohesion": round(float(sims[mask].mean()), 4),
        })
    return sorted(out, key=lambda row: row["cohesion"])


def compare(native: dict, pivot: dict) -> dict:
    """Native versus translate-to-English, and which one actually won.

    Two things decide this, in order.

    First, collapse. A pivot index over single common nouns does not retrieve
    translations better — a dictionary turns `eau`, `agua` and `Wasser` all
    into the literal string `water`, the model returns one vector for one
    string, and six documents become one. Every score goes to 1.0 because
    there is nothing left to tell apart. That is worth knowing and it is not a
    win, so when most of the grid has collapsed the verdict says so instead of
    reading out arithmetic.

    Second, the scores, in the order a retrieval system cares about them:
    retrieval accuracy, then how cleanly meaning separates from language, then
    the raw margin. It stops at the first one that moves by more than a
    hundredth, because on a grid this size a hundredth is one pair changing
    places.
    """
    if not (native.get("measurable") and pivot.get("measurable")):
        return {
            "verdict": "incomparable",
            "decided_on": "coverage",
            "note": (
                "One of the two indexes could not embed enough of the word set "
                "to measure. For GloVe that is the expected outcome in the "
                "native column and the whole point of the pivot one: "
                "translation is what gives a 2014 English word model a "
                "multilingual index at all."
            ),
        }

    checks = [
        ("translation_p_at_1", "retrieval"),
        ("meaning_beats_language", "separation"),
        ("gap", "margin"),
    ]
    deltas = {
        key: round((pivot.get(key) or 0.0) - (native.get(key) or 0.0), 4)
        for key, _ in checks
    }

    collapsed = pivot.get("collapsed_concepts") or []
    total = pivot.get("concepts") or 0
    native_collapsed = native.get("collapsed_concepts") or []
    # Half the grid is the line. Below it there is still a real comparison
    # happening among the concepts that survived; above it the pivot index is
    # mostly one vector per concept and the numbers are a tautology.
    mostly_collapsed = (
        bool(collapsed) and total and len(collapsed) * 2 >= total
        and not native_collapsed
    )

    if mostly_collapsed:
        return {
            "verdict": "pivot, definitionally",
            "decided_on": "collapse",
            "deltas": deltas,
            "collapsed_concepts": collapsed,
            "note": (
                f"{len(collapsed)} of {total} concepts came out of the "
                "translator as one identical string, so the pivot index holds "
                "a single vector where the native index holds several. Its "
                "perfect scores are arithmetic rather than skill — there is "
                "nothing left to retrieve across. The finding is real and it "
                "cuts both ways: translation does erase the language, and it "
                "erases the difference between the documents with it, which is "
                "perfect recall bought with precision nobody counted. The "
                "sentence grid is where this comparison has something to "
                "compare, because translations of a sentence are not "
                "byte-identical."
            ),
        }

    winner, decided_on = "tie", "retrieval"
    for key, label in checks:
        if abs(deltas[key]) > 0.01:
            winner = "pivot" if deltas[key] > 0 else "native"
            decided_on = label
            break

    note = ""
    if collapsed:
        note = (
            f"{len(collapsed)} of {total} concepts collapsed to one identical "
            "translation, which flatters the pivot column on those concepts. "
            "The verdict is drawn from the whole grid, so read it with that in "
            "mind."
        )
    elif winner == "tie":
        note = (
            "The two indexes score the same. That is a result rather than a "
            "non-result: the pivot column bought nothing here, and it cost a "
            "translation call for every document and every query."
        )

    return {
        "verdict": winner,
        "decided_on": decided_on,
        "deltas": deltas,
        "collapsed_concepts": collapsed,
        "note": note,
    }
