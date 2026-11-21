"""The HTTP surface.

Two endpoints carry the application. `/api/meta` tells the UI what exists —
which sources have keys, which methods are installed, what the presets are —
so the frontend ships no duplicate copy of that knowledge. `/api/scene` does
everything else in one round trip: embed whatever is new, store it, project all
of it, measure how badly the projection distorted things, and compute the true
similarities. One call per change to the word list keeps the UI simple and the
picture consistent, because the points and the numbers beside them always come
from the same request.
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    AlignmentIn,
    AlignmentOut,
    AnalogyOut,
    ArithmeticIn,
    ArithmeticOut,
    ConceptOut,
    DimensionsIn,
    DimensionsOut,
    EmbedOut,
    IndexOut,
    LanguageOut,
    LexiconOut,
    MetaOut,
    MethodOut,
    NeighboursOut,
    PivotOut,
    PointOut,
    PresetOut,
    SceneIn,
    SceneOut,
    SourceOut,
    TermOut,
    TranslationOut,
    WordsIn,
)
from app.data.glossary import TERMS
from app.data.multilingual import LEXICONS, Lexicon, lexicon_by_id
from app.data.presets import ANALOGIES, PRESETS, SAMPLER, all_preset_words
from app.services import alignment, dimensions, similarity, store, translate
from app.services.embedding import (
    DEFAULT_SOURCE_ID,
    SOURCES,
    SOURCES_BY_ID,
    EmbeddingError,
    Source,
    base_of,
    embed,
    glove,
    key_for,
    pivot_of,
    source_status,
)
from app.services.projection import (
    DEFAULT_METHOD_ID,
    METHODS,
    METHODS_BY_ID,
    ProjectionError,
    cluster,
    project,
    umap_available,
)
from app.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["embeddings"])


def _resolve_source(source_id: str) -> Source:
    source = SOURCES_BY_ID.get(source_id)
    if source is None:
        raise HTTPException(404, f"No embedding source called {source_id!r}")
    ok, why = source_status(source)
    if not ok:
        # 503 rather than 400: the request is valid, the server cannot serve it
        # yet, and the reason is actionable.
        raise HTTPException(503, f"{source.label} is unavailable — {why}")
    return source


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@router.get("/meta", response_model=MetaOut)
def meta() -> MetaOut:
    sources: list[SourceOut] = []
    for source in SOURCES:
        ok, why = source_status(source)
        sources.append(
            SourceOut(
                id=source.id,
                label=source.label,
                provider=source.provider,
                model=source.model,
                dims=source.dims,
                blurb=source.blurb,
                supports_phrases=source.supports_phrases,
                good_at_analogies=source.good_at_analogies,
                cross_lingual=source.cross_lingual,
                cross_lingual_note=source.cross_lingual_note,
                languages=source.languages,
                pivot=source.pivot,
                base=source.base,
                available=ok,
                unavailable_reason=why,
                # Counting means opening the collection, which creates it. That
                # is harmless and saves the UI a second call.
                indexed=store.count(source) if ok else 0,
                vocab_size=glove.vocab_size if source.provider == "glove" and ok else 0,
            )
        )

    has_umap = umap_available()
    methods = [
        MethodOut(
            id=m.id,
            label=m.label,
            blurb=m.blurb,
            detail=m.detail,
            params=m.params,
            deterministic=m.deterministic,
            min_points=m.min_points,
            available=has_umap or m.id != "umap",
            unavailable_reason=(
                "" if has_umap or m.id != "umap" else "umap-learn is not installed"
            ),
        )
        for m in METHODS
    ]

    # Which sources a word set may be used with, worked out here rather than
    # stored: a set of sentences needs a model that embeds sentences, and the
    # list of those changed three times while this feature was being written.
    phrase_sources = [s.id for s in SOURCES if s.supports_phrases]

    return MetaOut(
        sources=sources,
        methods=methods,
        presets=[
            PresetOut(
                id=p.id,
                label=p.label,
                blurb=p.blurb,
                look_for=p.look_for,
                words=p.words,
                sources=phrase_sources if p.requires_phrases else [],
                lexicon=p.lexicon,
            )
            for p in PRESETS
        ],
        analogies=[
            AnalogyOut(
                label=a.label,
                terms=[{"word": w, "sign": s} for w, s in a.terms],
                expect=a.expect,
                note=a.note,
            )
            for a in ANALOGIES
        ],
        lexicons=[
            LexiconOut(
                id=lexicon.id,
                label=lexicon.label,
                languages=[
                    LanguageOut(
                        code=language.code,
                        label=language.label,
                        endonym=language.endonym,
                        script=language.script,
                    )
                    for language in lexicon.languages
                ],
                concepts=[
                    ConceptOut(id=concept.id, gloss=concept.gloss)
                    for concept in lexicon.concepts
                ],
            )
            for lexicon in LEXICONS
        ],
        glossary=[
            TermOut(
                id=t.id,
                label=t.label,
                short=t.short,
                detail=t.detail,
                topic=t.topic,
            )
            for t in TERMS
        ],
        sampler=SAMPLER,
        default_source=DEFAULT_SOURCE_ID,
        default_method=DEFAULT_METHOD_ID,
        max_words=settings.max_words_per_projection,
    )


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


async def _ensure_embedded(
    source: Source, words: list[str]
) -> tuple[dict[str, np.ndarray], list[str], list[str], list[str]]:
    """Vectors for `words`, embedding and storing whatever is not cached yet.

    Returns (vectors by word, cached words, newly embedded words, missing).
    The cached/new split is surfaced in the response so the UI can say what a
    keystroke actually cost — the difference between a Chroma read and a
    billable API call is worth showing in a teaching tool.
    """
    keys: list[str] = []
    for word in words:
        key = key_for(source, word)
        if key and key not in keys:
            keys.append(key)

    have = store.fetch(source, keys)
    cached = [k for k in keys if k in have]
    todo = [k for k in keys if k not in have]

    newly: list[str] = []
    missing: list[str] = []
    if todo:
        try:
            vectors, missing = await embed(source, todo)
        # TranslationError too: a "translated first" source raises it from
        # inside embed() for a word the built-in sets do not cover, and with no
        # OPENAI_KEY its message — which says exactly what to do — would
        # otherwise surface as a bare "Internal Server Error".
        except (EmbeddingError, translate.TranslationError) as exc:
            raise HTTPException(502, str(exc)) from exc
        embedded = [k for k in todo if k not in missing]
        if len(embedded) != vectors.shape[0]:
            # Only GloVe can drop inputs, and it reports them in `missing`; if
            # the counts still disagree something is wrong with the provider
            # rather than the request.
            raise HTTPException(
                500,
                f"{source.label} returned {vectors.shape[0]} vectors for "
                f"{len(embedded)} inputs",
            )
        if embedded:
            store.upsert(source, embedded, vectors)
            have.update({w: v for w, v in zip(embedded, vectors)})
            newly = embedded
    return have, cached, newly, missing


@router.post("/embed", response_model=EmbedOut)
async def embed_words(payload: WordsIn) -> EmbedOut:
    source = _resolve_source(payload.source)
    _, cached, newly, missing = await _ensure_embedded(source, payload.words)
    return EmbedOut(
        source=source.id,
        embedded=newly,
        cached=cached,
        missing=missing,
        count=store.count(source),
    )


@router.get("/index", response_model=IndexOut)
def get_index(source: str = Query(...)) -> IndexOut:
    src = _resolve_source(source)
    words = store.words(src)
    return IndexOut(source=src.id, count=len(words), words=words)


@router.post("/index/seed", response_model=EmbedOut)
async def seed_index(source: str = Query(...)) -> EmbedOut:
    """Embed every preset word at once.

    Worth a button of its own for the OpenAI sources, where neighbour and
    analogy queries can only answer with words that are already in the index.
    Seeding turns "the nearest of the four words you typed" into "the nearest
    of 254", which is the difference between the feature working and not.
    """
    src = _resolve_source(source)
    _, cached, newly, missing = await _ensure_embedded(src, all_preset_words())
    return EmbedOut(
        source=src.id,
        embedded=newly,
        cached=cached,
        missing=missing,
        count=store.count(src),
    )




@router.delete("/index", response_model=IndexOut)
def delete_words(payload: WordsIn) -> IndexOut:
    src = _resolve_source(payload.source)
    if payload.words:
        store.delete(src, payload.words)
    else:
        store.clear(src)
    words = store.words(src)
    return IndexOut(source=src.id, count=len(words), words=words)


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------


@router.post("/scene", response_model=SceneOut)
async def scene(payload: SceneIn) -> SceneOut:
    source = _resolve_source(payload.source)
    method = METHODS_BY_ID.get(payload.method)
    if method is None:
        raise HTTPException(404, f"No projection method called {payload.method!r}")

    vectors_by_word, _, newly, missing = await _ensure_embedded(source, payload.words)

    # Preserve the order the client sent. The word list is the user's, and
    # re-ordering it would scramble their colour legend and their chips.
    words: list[str] = []
    for word in payload.words:
        key = key_for(source, word)
        if key in vectors_by_word and key not in words:
            words.append(key)

    notes: list[str] = []
    if missing:
        notes.append(
            f"{source.label} has no vector for {', '.join(missing[:5])}"
            + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            + ("; GloVe only knows single words from its 50k vocabulary."
               if source.provider == "glove" else ".")
        )

    if not words:
        raise HTTPException(
            400,
            "No words could be embedded. "
            + (notes[0] if notes else "Add some words to get started."),
        )

    matrix_in = np.vstack([vectors_by_word[w] for w in words])

    try:
        coords, projection_meta = project(
            source_id=source.id,
            method_id=method.id,
            words=words,
            vectors=matrix_in,
            params=payload.params,
            anchor=payload.anchor,
        )
    except ProjectionError as exc:
        raise HTTPException(400, str(exc)) from exc

    labels = (
        cluster(matrix_in, payload.clusters)
        if payload.clusters and payload.clusters > 1
        else [0] * len(words)
    )

    similarities = similarity.cosine_matrix(matrix_in)

    # Neighbours within the scene, for the edges drawn on hover. Distinct from
    # /api/neighbours, which searches the whole vocabulary — here the question
    # is "of the words in front of me, which are closest", and the answer has
    # to be restricted to them or the edges would point off-screen.
    neighbours: dict[str, list[dict]] = {}
    if payload.neighbour_k > 0 and len(words) > 1:
        k = min(payload.neighbour_k, len(words) - 1)
        for i, word in enumerate(words):
            row = similarities[i].copy()
            row[i] = -np.inf
            order = np.argpartition(-row, k - 1)[:k]
            order = order[np.argsort(-row[order])]
            neighbours[word] = [
                {"word": words[j], "similarity": round(float(similarities[i][j]), 4)}
                for j in order
            ]

    if projection_meta.get("clamped"):
        notes.append(str(projection_meta["clamped"]))

    # Words that share a vector exactly, and what this particular projection
    # did about it. Both halves matter. The first explains the clumps; the
    # second is the more interesting warning, because a method that spreads
    # identical vectors apart is drawing a difference that is not in the data,
    # and it can do that while reporting a high trustworthiness — UMAP gives
    # all 48 words of the translated translations grid their own position and
    # scores 0.94 for it.
    identical = similarity.identical_groups(words, matrix_in)
    if identical:
        drawn = len({(round(c[0], 5), round(c[1], 5), round(c[2], 5)) for c in coords})
        notes.append(
            similarity.describe_identical_groups(
                identical,
                len(words),
                method.label,
                drawn,
                translated=bool(source.pivot),
            )
        )

    return SceneOut(
        source=source.id,
        method=method.id,
        labels=_labels_for(source, payload.lexicon, words),
        points=[
            PointOut(
                word=word,
                x=round(float(coords[i][0]), 5),
                y=round(float(coords[i][1]), 5),
                z=round(float(coords[i][2]), 5),
                cluster=labels[i],
            )
            for i, word in enumerate(words)
        ],
        projection=projection_meta,
        neighbours=neighbours,
        top_pairs=similarity.top_pairs(words, similarities),
        # Rounded to four places: the matrix is O(n^2) on the wire and nobody
        # reads the fifth decimal of a cosine.
        matrix=(
            [[round(float(v), 4) for v in row] for row in similarities]
            if payload.include_matrix
            else None
        ),
        missing=missing,
        newly_embedded=newly,
        notes=notes,
    )


def _labels_for(
    source: Source, lexicon_id: str, words: list[str]
) -> dict[str, dict[str, str]]:
    """The language and concept of every word on screen, keyed as the points are.

    Keyed by `key_for`, not by the string the grid holds: on a cased source the
    point is labelled "Wasser" and on GloVe it is labelled "wasser", and a
    label map that used the grid's spelling would silently fail to match on one
    of them.
    """
    if not lexicon_id:
        return {}
    lexicon = lexicon_by_id(lexicon_id)
    if lexicon is None:
        raise HTTPException(404, f"No word grid called {lexicon_id!r}")
    on_screen = set(words)
    labels: dict[str, dict[str, str]] = {}
    for word, language, concept in lexicon.entries():
        key = key_for(source, word)
        if key in on_screen:
            labels[key] = {"language": language, "concept": concept}
    return labels


# ---------------------------------------------------------------------------
# Cross-lingual alignment
# ---------------------------------------------------------------------------


async def _measure(source: Source, lexicon: Lexicon) -> tuple[dict, list[str]]:
    """Embed a whole grid under one source and run the alignment report.

    Words the source cannot embed are dropped and reported rather than
    substituted. That matters most for GloVe, where two thirds of the
    `scripts` grid has no vector at all — a report computed over the third
    that survived would be a flattering answer to a question nobody asked, so
    `alignment.evaluate` refuses to score a grid that has collapsed and says
    why instead.
    """
    words = lexicon.words()
    vectors_by_word, _, _, missing = await _ensure_embedded(source, words)

    entries: list[alignment.Entry] = []
    matrix: list[np.ndarray] = []
    seen: set[str] = set()
    for word, language, concept in lexicon.entries():
        key = key_for(source, word)
        if key not in vectors_by_word or key in seen:
            continue
        seen.add(key)
        entries.append(alignment.Entry(text=key, language=language, concept=concept))
        matrix.append(vectors_by_word[key])

    if not entries:
        return {
            "measurable": False,
            "reason": f"{source.label} could not embed any of this word set.",
            "words": 0,
            "concepts": 0,
            "languages": 0,
        }, missing
    return alignment.evaluate(entries, np.vstack(matrix)), missing


@router.post("/alignment", response_model=AlignmentOut)
async def measure_alignment(payload: AlignmentIn) -> AlignmentOut:
    """How much this model sorts by meaning rather than by language.

    Computed on the full vectors, so it is a property of the model rather than
    of the picture — flipping to t-SNE does not move these numbers, which is
    the entire reason they are worth printing next to a t-SNE plot.
    """
    source = _resolve_source(payload.source)
    lexicon = lexicon_by_id(payload.lexicon)
    if lexicon is None:
        raise HTTPException(404, f"No word grid called {payload.lexicon!r}")
    if lexicon.requires_phrases and not source.supports_phrases:
        raise HTTPException(
            400,
            f"{source.label} has no vector for a sentence, and averaging the "
            "words in one would make this number meaningless rather than low.",
        )

    report, missing = await _measure(source, lexicon)
    notes: list[str] = []
    if missing:
        notes.append(
            f"{source.label} has no vector for {len(missing)} of "
            f"{len(lexicon.words())} words: {', '.join(missing[:6])}"
            + (f" and {len(missing) - 6} more" if len(missing) > 6 else "")
        )
    if source.pivot:
        notes.append(
            "Every word here was translated to English before being embedded, "
            "so this measures the translator and an English index together."
        )
    return AlignmentOut(
        source=source.id,
        lexicon=lexicon.id,
        report=report,
        missing=missing,
        notes=notes,
    )


@router.post("/pivot", response_model=PivotOut)
async def compare_pivot(payload: AlignmentIn) -> PivotOut:
    """One model, both index designs, side by side.

    Give it a native source and it measures that source and its
    translate-to-English twin over the same grid, then says which won and on
    what. This is the endpoint that answers "should I translate my documents
    or trust the model", and it answers it per model, because the answer is
    not the same for all of them.
    """
    native = _resolve_source(base_of(_resolve_source(payload.source)).id)
    twin = pivot_of(native)
    if twin is None:
        raise HTTPException(500, f"{native.id} has no pivot twin")
    lexicon = lexicon_by_id(payload.lexicon)
    if lexicon is None:
        raise HTTPException(404, f"No word grid called {payload.lexicon!r}")
    if lexicon.requires_phrases and not native.supports_phrases:
        raise HTTPException(
            400,
            f"{native.label} has no vector for a sentence. The comparison needs "
            "both halves to be meaningful, and this half would not be.",
        )

    try:
        native_report, native_missing = await _measure(native, lexicon)
        pivot_report, pivot_missing = await _measure(twin, lexicon)
    except translate.TranslationError as exc:
        raise HTTPException(502, str(exc)) from exc

    translations = await translate.to_english(lexicon.words())
    notes: list[str] = []
    if native_missing:
        notes.append(
            f"Natively, {native.label} has no vector for {len(native_missing)} of "
            f"{len(lexicon.words())} words — which is the first half of the "
            "answer, before any similarity is computed."
        )
    if pivot_missing:
        notes.append(
            f"After translation, {len(pivot_missing)} words still have no "
            "vector: the English word is outside the model's vocabulary too."
        )
    guessed = [t for t in translations.values() if t.via == "model"]
    if guessed:
        notes.append(
            f"{len(guessed)} translations came from the translator rather than "
            "the word set's own glossary, so they carry its errors."
        )

    return PivotOut(
        lexicon=lexicon.id,
        native_source=native.id,
        pivot_source=twin.id,
        native=native_report,
        pivot=pivot_report,
        comparison=alignment.compare(native_report, pivot_report),
        translations=[
            TranslationOut(text=t.source_text, english=t.english, via=t.via)
            for t in translations.values()
        ],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@router.post("/dimensions", response_model=DimensionsOut)
async def word_dimensions(payload: DimensionsIn) -> DimensionsOut:
    """Which dimensions single a word out, and which words share them.

    The counterweight to the raw projection method. Raw mode shows that
    *drawing* three dimensions is hopeless; this shows that the set of
    dimensions a word is unusual on is nonetheless shared by its translations,
    and reports the rank correlation between that overlap and cosine so the
    claim is checkable rather than suggested.

    Scores are relative to the comparison set the caller sent, which is
    normally the words on screen. That is the right frame for "what makes this
    word different from these words" and the wrong one for any claim about the
    model's vocabulary at large, so the response says which it made.
    """
    source = _resolve_source(payload.source)

    requested = payload.words or [payload.word]
    if payload.word not in requested:
        requested = [payload.word, *requested]
    vectors_by_word, _, _, missing = await _ensure_embedded(source, requested)

    words: list[str] = []
    rows: list[np.ndarray] = []
    for word in requested:
        key = key_for(source, word)
        if key in vectors_by_word and key not in words:
            words.append(key)
            rows.append(vectors_by_word[key])

    target = key_for(source, payload.word)
    if target not in words:
        raise HTTPException(
            404,
            f"{source.label} has no vector for {payload.word!r}"
            + (f" — nor for {len(missing) - 1} other words here" if len(missing) > 1 else ""),
        )

    try:
        report = dimensions.profile(
            words, np.vstack(rows), target, top_k=payload.top_k
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    notes: list[str] = []
    if len(words) < 4:
        notes.append(
            f"Only {len(words)} words to compare against, so the deviation "
            "scores and the overlap correlation are both weak. Load a word set."
        )
    if report["hub_dimensions"]:
        notes.append(
            f"{len(report['hub_dimensions'])} of the largest dimensions here are "
            "large for most words in this set, so they say more about "
            f"{source.label} than about {payload.word!r}. The ranking by "
            "deviation excludes them."
        )
    return DimensionsOut(source=source.id, profile=report, notes=notes)


# ---------------------------------------------------------------------------
# Full-dimensional queries
# ---------------------------------------------------------------------------


@router.get("/neighbours", response_model=NeighboursOut)
async def neighbours(
    source: str = Query(...), word: str = Query(...), k: int = Query(10, ge=1, le=100)
) -> NeighboursOut:
    src = _resolve_source(source)
    if src.pivot:
        # "Translate the query too" is the pivot design, and this is where it
        # happens. Without this the word would have to already be in the index
        # for its own neighbour query to work.
        await _ensure_embedded(src, [word])
    try:
        hits, pool = similarity.neighbours_of(src, word, k)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    return NeighboursOut(
        word=key_for(src, word),
        neighbours=[
            {"word": h["word"], "similarity": round(float(h["similarity"]), 4)}
            for h in hits
        ],
        pool=pool,
    )


@router.post("/arithmetic", response_model=ArithmeticOut)
async def arithmetic(payload: ArithmeticIn) -> ArithmeticOut:
    source = _resolve_source(payload.source)

    # On OpenAI the terms have to be in the index before they can be added
    # together, and the user reached this via a preset button that assumes
    # nothing. Embedding them here is what makes those buttons work on a fresh
    # index.
    await _ensure_embedded(source, [w for w, _ in payload.terms])

    result = similarity.arithmetic(source, list(payload.terms), payload.topk)
    note = ""
    if not source.good_at_analogies:
        note = (
            f"{source.label} is a contrastively trained sentence encoder, not "
            "word2vec. Offset arithmetic is much weaker on it than on GloVe — "
            "treat a miss here as a property of the model, not a bug."
        )
    if result["unknown"]:
        note = (
            f"No vector for {', '.join(result['unknown'])}. "
            + note
        ).strip()
    return ArithmeticOut(
        result=[
            {"word": h["word"], "similarity": round(float(h["similarity"]), 4)}
            for h in result["result"]
        ],
        terms=[
            {
                "word": t["word"],
                "sign": t["sign"],
                "similarity_to_result": round(float(t["similarity_to_result"]), 4),
            }
            for t in result["terms"]
        ],
        unknown=result["unknown"],
        pool=result["pool"],
        norm=round(float(result.get("norm", 0.0)), 4),
        note=note,
    )
