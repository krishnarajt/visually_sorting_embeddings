"""Request and response shapes for the API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.settings import settings


class SourceOut(BaseModel):
    id: str
    label: str
    provider: str
    model: str
    dims: int
    blurb: str
    supports_phrases: bool
    good_at_analogies: bool
    # How the model came by its multilingual ability: none, incidental,
    # multilingual, aligned, or pivot. The UI groups the picker by it.
    cross_lingual: str = "none"
    cross_lingual_note: str = ""
    languages: int = 1
    # "en" for a translate-to-English twin, "" for a source that embeds the
    # text it is given. `base` names the native source a twin came from, so the
    # UI can render eight models and a switch rather than sixteen rows.
    pivot: str = ""
    base: str = ""
    available: bool
    unavailable_reason: str = ""
    indexed: int = 0
    vocab_size: int = 0


class MethodOut(BaseModel):
    id: str
    label: str
    blurb: str
    detail: str
    params: list[dict[str, Any]]
    deterministic: bool
    min_points: int
    available: bool
    unavailable_reason: str = ""


class PresetOut(BaseModel):
    id: str
    label: str
    blurb: str
    look_for: str
    words: list[str]
    # The sources this set can actually be used with, resolved from the
    # preset's `requires_phrases` against each source's `supports_phrases`.
    # Sent as a list rather than a flag so the UI does not have to redo the
    # reasoning.
    sources: list[str]
    # The concept-by-language grid these words came from, "" for a plain set.
    lexicon: str = ""


class AnalogyOut(BaseModel):
    label: str
    terms: list[dict[str, Any]]
    expect: str
    note: str


class LanguageOut(BaseModel):
    code: str
    label: str
    endonym: str
    script: str


class ConceptOut(BaseModel):
    id: str
    gloss: str


class LexiconOut(BaseModel):
    """A concept-by-language grid, as the UI needs to describe one.

    The words themselves are not here — they are in the matching PresetOut,
    because to every control in the UI a grid is just another word set. What
    is here is what the legend and the alignment panel need: the axes.
    """

    id: str
    label: str
    languages: list[LanguageOut]
    concepts: list[ConceptOut]


# ---------------------------------------------------------------------------
# The glossary
# ---------------------------------------------------------------------------


class TermOut(BaseModel):
    """One thing the UI can explain on hover.

    Served rather than hardcoded in the frontend for the reason /api/meta
    exists at all: the definition of trustworthiness belongs next to the code
    that computes trustworthiness, and a second copy in a React component is a
    second thing to keep true. `detail` is the longer form for the reference
    list; `short` is what fits in a tooltip.
    """

    id: str
    label: str
    short: str
    detail: str = ""
    # Roughly where it appears, so the reference list can be grouped rather
    # than alphabetical — nobody looks up "stress" by its initial letter.
    topic: str = ""


class MetaOut(BaseModel):
    """Everything the UI needs to render its controls, in one request."""

    sources: list[SourceOut]
    methods: list[MethodOut]
    presets: list[PresetOut]
    analogies: list[AnalogyOut]
    lexicons: list[LexiconOut]
    glossary: list[TermOut]
    sampler: list[str]
    default_source: str
    default_method: str
    max_words: int


class WordsIn(BaseModel):
    source: str
    words: list[str] = Field(default_factory=list)

    @field_validator("words")
    @classmethod
    def _cap(cls, value: list[str]) -> list[str]:
        if len(value) > settings.max_words_per_projection:
            raise ValueError(
                f"{len(value)} words is over the {settings.max_words_per_projection} limit"
            )
        return value


class SceneIn(BaseModel):
    """One request for a complete view: embed what is new, then project it all."""

    source: str
    words: list[str]
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    # 0 or 1 means "do not colour by cluster".
    clusters: int = 0
    # How many neighbour edges to return per point, for the hover graph.
    neighbour_k: int = 3
    # Where these words sat in the previous frame. Used to keep t-SNE and UMAP
    # from spinning the whole cloud when one word is added — see
    # projection._align.
    anchor: dict[str, list[float]] = Field(default_factory=dict)
    # Whether to include the full pairwise cosine matrix. It is O(n^2) on the
    # wire, so the heatmap panel asks for it and nothing else does.
    include_matrix: bool = True
    # The grid these words came from, if any. Set it and the response carries a
    # language and a concept for every point, which is what the colour-by-
    # language and colour-by-meaning modes are drawn from.
    lexicon: str = ""

    @field_validator("words")
    @classmethod
    def _cap(cls, value: list[str]) -> list[str]:
        if len(value) > settings.max_words_per_projection:
            raise ValueError(
                f"{len(value)} words is over the {settings.max_words_per_projection} limit"
            )
        return value


class PointOut(BaseModel):
    word: str
    x: float
    y: float
    z: float
    cluster: int = 0


class SceneOut(BaseModel):
    source: str
    method: str
    points: list[PointOut]
    # word -> {"language": "fr", "concept": "bread"}, present only when the
    # request named a lexicon. Keyed by the same word the points are, which for
    # a cased source is not the same string as the lexicon holds.
    labels: dict[str, dict[str, str]] = Field(default_factory=dict)
    # Quality metrics and method-specific readouts: trustworthiness,
    # distance_correlation, explained variance, stress, and so on.
    projection: dict[str, Any]
    # Nearest neighbours within the words on screen, for drawing edges.
    neighbours: dict[str, list[dict[str, Any]]]
    top_pairs: list[dict[str, Any]]
    matrix: list[list[float]] | None = None
    # Words the source could not embed at all (out of GloVe's vocabulary).
    missing: list[str] = Field(default_factory=list)
    # Words that were embedded by this request rather than served from Chroma.
    newly_embedded: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArithmeticIn(BaseModel):
    source: str
    # [["king", 1], ["man", -1], ["woman", 1]]
    terms: list[tuple[str, int]]
    topk: int = 8

    @field_validator("terms")
    @classmethod
    def _check(cls, value: list[tuple[str, int]]) -> list[tuple[str, int]]:
        if not value:
            raise ValueError("at least one term is needed")
        if len(value) > 8:
            raise ValueError("eight terms is plenty")
        for word, sign in value:
            if sign not in (1, -1):
                raise ValueError(f"sign for {word!r} must be +1 or -1")
        return value


class ArithmeticOut(BaseModel):
    result: list[dict[str, Any]]
    terms: list[dict[str, Any]]
    unknown: list[str]
    pool: str
    norm: float = 0.0
    note: str = ""


class NeighboursOut(BaseModel):
    word: str
    neighbours: list[dict[str, Any]]
    pool: str


class IndexOut(BaseModel):
    source: str
    count: int
    words: list[str]


class EmbedOut(BaseModel):
    source: str
    embedded: list[str]
    cached: list[str]
    missing: list[str]
    count: int


# ---------------------------------------------------------------------------
# Cross-lingual alignment
# ---------------------------------------------------------------------------


class AlignmentIn(BaseModel):
    source: str
    lexicon: str


class AlignmentOut(BaseModel):
    source: str
    lexicon: str
    # The full report from app/services/alignment.py. Left as a dict for the
    # same reason SceneOut.projection is: it is a bag of measurements whose
    # membership is the module's business, and mirroring every key here would
    # only create a second place for them to disagree.
    report: dict[str, Any]
    missing: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TranslationOut(BaseModel):
    text: str
    english: str
    # "glossary", "cache" or "model" — which translator produced this, because
    # a pivot index is only as good as its translator and the user is entitled
    # to know which words were guessed at.
    via: str


class PivotOut(BaseModel):
    """One model measured both ways: as itself, and behind a translator."""

    lexicon: str
    native_source: str
    pivot_source: str
    native: dict[str, Any]
    pivot: dict[str, Any]
    # Which index won, on which measure, and by how much.
    comparison: dict[str, Any]
    translations: list[TranslationOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


class DimensionsIn(BaseModel):
    source: str
    # The word to profile, and the words to profile it against. The comparison
    # set is the caller's because the z-scores are relative to it — "unusual
    # among these words" is a different claim from "unusual in the model", and
    # the response says which it made.
    word: str
    words: list[str] = Field(default_factory=list)
    top_k: int = 8

    @field_validator("top_k")
    @classmethod
    def _sane(cls, value: int) -> int:
        if not 1 <= value <= 64:
            raise ValueError("top_k has to be between 1 and 64")
        return value


class DimensionsOut(BaseModel):
    source: str
    profile: dict[str, Any]
    notes: list[str] = Field(default_factory=list)
