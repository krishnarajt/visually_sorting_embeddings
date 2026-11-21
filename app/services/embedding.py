"""The embedding sources, and the rule that keeps them apart.

Eight sources, four providers, and one hard constraint: a 1536-d OpenAI vector
and a 300-d GloVe vector for the word "king" describe the same word in
completely unrelated coordinate systems. Cosine between them is not a smaller
similarity, it is a type error. So each source gets its own Chroma collection
(see app/services/store.py) and no code path ever mixes them.

The four providers differ in kind, not just in size:

  GloVe   — a 2014 count-based *word* model. One vector per token, nothing for
            out-of-vocabulary words, and the analogy arithmetic it was famous
            for actually works. Not normalised on disk. English only.
  OpenAI  — a contrastively trained *sentence* encoder behind an HTTP API.
            Embeds anything you can type, including a paragraph. Vectors come
            back L2-normalised. Multilingual, without saying how.
  Cohere  — also an HTTP API, and the only provider here that asks what the
            vector is *for* before it makes one: embed v3 requires an
            `input_type`, and the same string embedded for search and for
            clustering comes back as two different vectors. See `Source.
            input_type` and `embed_cohere`.
  onnx    — four transformer encoders run locally on ONNX Runtime, chosen for
            how they came by their multilingual ability rather than for
            variety. See app/services/encoders.py.

They are all worth having precisely because they disagree. `cross_lingual`
below is the axis that matters for the translations word sets: it records
whether a model was ever *told* that two strings mean the same thing in
different languages, which turns out to predict the numbers in
app/services/alignment.py almost perfectly.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from app.services.encoders import encoder_for
from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    id: str
    label: str
    provider: str
    model: str
    dims: int
    blurb: str
    # GloVe has a vector per token and nothing for a sentence. Asking it for
    # one averages the words, which is a real technique and a bad demo, so the
    # UI greys out the sentence preset instead of quietly lying.
    supports_phrases: bool
    # Whether vector arithmetic is expected to land somewhere sensible.
    good_at_analogies: bool
    # Whether the model distinguishes "Wasser" from "wasser". GloVe's
    # vocabulary is lowercase, so its keys have to be folded; the transformer
    # models are cased and folding them costs real accuracy — on the
    # translations set it drops minilm-multilingual's translation retrieval
    # from 100% to 92%, because German capitalises its nouns and a model that
    # was trained on that spelling is being shown a different string. So the
    # key a vector is stored under is a property of the source, not a global.
    # See key_for(), and store.py for why per-source keys are safe: each
    # source owns its own collection.
    cased: bool = True
    # Cohere's embed v3 takes a required `input_type`, and it is not a hint:
    # the same string asked for as a `search_document` and as a `clustering`
    # input comes back as two different vectors. Which one is correct depends
    # on what the numbers are for, so it is a property of the source — fixed,
    # like the pooling in encoders.py and e5's `query: ` prefix — rather than a
    # setting. Two input types in one collection would be two subtly different
    # geometries in one index, and nothing would ever raise about it. Empty for
    # every provider that does not have the concept.
    input_type: str = ""
    # How the model came by whatever cross-lingual ability it has. One of:
    #   none        — trained on one language.
    #   incidental  — trained on one language's text, which happened to
    #                 contain foreign words.
    #   multilingual — trained on many languages, never told which strings are
    #                 translations of each other.
    #   aligned     — trained with translation pairs as an explicit objective.
    # The word sets in app/data/multilingual.py exist to make the difference
    # between the last two visible.
    cross_lingual: str = "none"
    cross_lingual_note: str = ""
    # Languages the model is published as covering. 0 where the publisher does
    # not give a number.
    languages: int = 1
    # "" for a source that embeds the text it is given, or "en" for the
    # translate-to-English pivot twin, which embeds the *English translation*
    # of the text and files it under the original. Every model here has both,
    # because "translate first, then use an English index" is the other way to
    # build multilingual search and the comparison is the point. See
    # app/services/translate.py.
    pivot: str = ""
    # The native source a pivot twin was derived from, "" if this is one.
    base: str = ""


# Ordered along the axis the translations word sets are about: from a model
# that was never shown a second language to one whose whole training objective
# was to put translations in the same place.
NATIVE_SOURCES: list[Source] = [
    Source(
        id="glove-300",
        label="GloVe 6B 300d",
        provider="glove",
        model="glove.6B.300d",
        dims=300,
        blurb=(
            "Stanford, 2014, trained on 6 billion tokens of Wikipedia. Single "
            "words only — but this is the model king − man + woman was discovered "
            "on, and the arithmetic still works."
        ),
        supports_phrases=False,
        good_at_analogies=True,
        # The .npz built by fetch_glove.py is lowercase throughout, so an
        # unfolded key could never match anything.
        cased=False,
        cross_lingual="incidental",
        cross_lingual_note=(
            "English Wikipedia and Gigaword, nothing else. It has a vector for "
            "'agua' only because English text sometimes contains 'agua', and "
            "that vector is placed by the English company the word keeps — not "
            "by what it means in Spanish. Half the translations word set is "
            "outside its vocabulary entirely."
        ),
        languages=1,
    ),
    Source(
        id="openai-3-small",
        label="OpenAI text-embedding-3-small",
        provider="openai",
        model="text-embedding-3-small",
        dims=1536,
        blurb=(
            "The default modern choice. 1536 dimensions, understands phrases and "
            "sentences, costs about $0.02 per million tokens."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="multilingual",
        cross_lingual_note=(
            "Multilingual, and OpenAI does not publish the recipe or the "
            "language count. It retrieves translations reliably; it also keeps "
            "a visible amount of the language itself in the vector."
        ),
        languages=0,
    ),
    Source(
        id="openai-3-large",
        label="OpenAI text-embedding-3-large",
        provider="openai",
        model="text-embedding-3-large",
        dims=3072,
        blurb=(
            "Twice the dimensions of 3-small and measurably better at retrieval. "
            "Twice as many numbers to flatten into three, too."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="multilingual",
        cross_lingual_note=(
            "The larger sibling, and — on the translations set — the more "
            "language-sensitive of the two. Bigger is not the same as more "
            "language-agnostic."
        ),
        languages=0,
    ),
    Source(
        id="cohere-embed-v3-multilingual",
        label="Cohere embed-multilingual-v3.0",
        provider="cohere",
        model="embed-multilingual-v3.0",
        dims=1024,
        blurb=(
            "100+ languages behind an HTTP API, L2-normalised at 1024 "
            "dimensions, and the only model here that asks what the vectors "
            "are for: these are clustering embeddings, not search ones."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="multilingual",
        cross_lingual_note=(
            "Cohere documents cross-lingual retrieval as a headline capability "
            "— a Chinese query against Finnish documents — but publishes no "
            "paper and no training recipe, so on evidence it files with the "
            "merely multilingual models rather than beside LaBSE and MiniLM, "
            "whose objectives are written down. Measured, it sits exactly "
            "there, and not at the top: its mean same-meaning cosine of 0.793 "
            "is below LaBSE's and e5's, and its gap of +0.174 is fifth of "
            "eight. What it leads on is separability among the merely "
            "multilingual — concept silhouette +0.480, AUC 0.943, best-cutoff "
            "F1 0.917 — while its silhouette *by language* stays *positive* at "
            "+0.052, where every translation-trained model goes negative. It "
            "retrieves every translation without having been taught to forget "
            "which language it was in, which is a property rather than a "
            "virtue: it is what you want for language-aware search and the "
            "wrong thing for language-blind search."
        ),
        languages=100,
        # Every number this app computes is a cosine between two items drawn
        # from the same pool — the similarity matrix, k-means, the silhouette,
        # the translation retrieval. That is the symmetric case, and Cohere's
        # own guidance is that `search_query`/`search_document` are for the
        # asymmetric one, where a three-word query and a three-paragraph
        # document are trained to be read against each other. Asking for search
        # embeddings and then measuring them against each other would be using
        # the model off-distribution in a way no error would report.
        input_type="clustering",
    ),
    Source(
        id="e5-multilingual-small",
        label="multilingual-e5-small",
        provider="onnx",
        model="intfloat/multilingual-e5-small",
        dims=384,
        blurb=(
            "118M parameters, 100 languages, runs on the CPU in front of you. A "
            "general multilingual retriever rather than a translation model."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="multilingual",
        cross_lingual_note=(
            "Contrastively pretrained on a billion mined multilingual pairs, "
            "then fine-tuned for retrieval. Never given a translation "
            "dictionary. It ranks translations first and still puts every pair "
            "of strings above 0.8 — which is why retrieval accuracy and the "
            "same-meaning gap have to be read together."
        ),
        languages=100,
    ),
    Source(
        id="minilm-multilingual",
        label="paraphrase-multilingual-MiniLM-L12-v2",
        provider="onnx",
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dims=384,
        blurb=(
            "118M parameters, 50 languages, 384 dimensions. Distilled so that a "
            "translation lands where the English original already was."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="aligned",
        cross_lingual_note=(
            "Multilingual knowledge distillation (Reimers & Gurevych, 2020): a "
            "monolingual English teacher embedded a sentence, and the student "
            "was trained to put that sentence *and its translations* at the "
            "same coordinates. Cross-lingual alignment is not a side effect "
            "here, it is the loss function."
        ),
        languages=50,
    ),
    Source(
        id="bge-m3",
        label="BGE-M3",
        provider="onnx",
        model="BAAI/bge-m3",
        dims=1024,
        blurb=(
            "568M parameters, 100+ languages, an 8192-token context. The "
            "current strong multilingual retriever, and the largest model here."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="aligned",
        cross_lingual_note=(
            "BAAI, 2024. Trained on a mixture that includes mined parallel "
            "text with cross-lingual retrieval as a stated objective, so it "
            "belongs with the aligned models rather than the merely "
            "multilingual ones — but it is a retrieval model first and a "
            "translation model second, which is the interesting difference "
            "between it and LaBSE."
        ),
        languages=100,
    ),
    Source(
        id="labse",
        label="LaBSE",
        provider="onnx",
        model="sentence-transformers/LaBSE",
        dims=768,
        blurb=(
            "471M parameters, 109 languages. Trained on 6 billion translation "
            "pairs — the reference model for finding a sentence's translation."
        ),
        supports_phrases=True,
        good_at_analogies=False,
        cross_lingual="aligned",
        cross_lingual_note=(
            "Language-agnostic BERT Sentence Embedding (Feng et al., 2020). "
            "Trained with an additive-margin softmax over 6B translation pairs, "
            "which pushes a sentence and its translation together and every "
            "other pair apart. If any model here should make language "
            "invisible, it is this one."
        ),
        languages=109,
    ),
]

def _pivot_twin(base: Source) -> Source:
    """The translate-to-English twin of a source.

    Same model, same dimensionality, same everything — the only difference is
    that the text handed to the model is an English translation rather than
    the original. That difference is enough to make it a separate source with
    a separate collection, because the vector filed under "собака" here is the
    vector of "dog", and mixing those two indexes would be exactly the type
    error the one-collection-per-source rule exists to prevent.
    """
    return Source(
        id=f"{base.id}-via-en",
        label=f"{base.label}, translated first",
        provider=base.provider,
        model=base.model,
        dims=base.dims,
        blurb=(
            f"Every word translated to English, then embedded by {base.label}. "
            "The query gets translated too, so the whole index is monolingual "
            "and any English model can serve it."
        ),
        supports_phrases=base.supports_phrases,
        good_at_analogies=base.good_at_analogies,
        cased=base.cased,
        input_type=base.input_type,
        cross_lingual="pivot",
        cross_lingual_note=(
            "Cross-lingual retrieval without a cross-lingual model: translate "
            "the documents to English, translate the query to English, and let "
            "an English index do the work. The model's own language coverage "
            "stops mattering — what matters instead is the translator, and "
            "everything the original wording carried that a translation does "
            "not."
        ),
        languages=base.languages,
        pivot="en",
        base=base.id,
    )


# Both halves of the comparison, native first. The UI renders the eight models
# as a list and the pivot as a switch, rather than showing sixteen rows of
# which half are the same eight models again.
PIVOT_SOURCES: list[Source] = [_pivot_twin(s) for s in NATIVE_SOURCES]
SOURCES: list[Source] = NATIVE_SOURCES + PIVOT_SOURCES

SOURCES_BY_ID: dict[str, Source] = {s.id: s for s in SOURCES}
DEFAULT_SOURCE_ID = "glove-300"


def base_of(source: Source) -> Source:
    """The native source behind a pivot twin, or the source itself."""
    return SOURCES_BY_ID[source.base] if source.base else source


def pivot_of(source: Source) -> Source | None:
    """The pivot twin of a native source, or None if given a twin already."""
    if source.pivot:
        return None
    return SOURCES_BY_ID.get(f"{source.id}-via-en")


class EmbeddingError(RuntimeError):
    """Raised when a source cannot produce vectors, with a message for the UI."""


# ---------------------------------------------------------------------------
# Text keys
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalise_text(text: str, fold_case: bool = True) -> str:
    """The canonical form of a piece of text, used as its id in Chroma.

    Whitespace-collapsed, and lowercased unless told otherwise, so "King"
    typed twice is one point and one API call rather than two of each. Kept
    deliberately gentle: stripping punctuation would merge "u.s." into "us",
    and the GloVe vocabulary distinguishes them.

    Almost nothing should call this directly — `key_for` picks `fold_case`
    from the source, which is the only way to get it right for a mixed set of
    cased and uncased models.
    """
    collapsed = _WS.sub(" ", text.strip())
    return collapsed.lower() if fold_case else collapsed


def key_for(source: Source, text: str) -> str:
    """The id `text` is stored and looked up under, for this source.

    Case folding is a property of the model, not of the app. GloVe's
    vocabulary is lowercase and an unfolded key would never match it; the
    transformer models are cased, and folding "Wasser" to "wasser" hands them
    a spelling German does not use. Measured on the translations word set that
    is not a rounding error — it costs minilm-multilingual two of its
    twenty-four translation retrievals.

    Two sources disagreeing about the key for the same string is safe because
    they never share an index: store.py gives each source its own Chroma
    collection.
    """
    return normalise_text(text, fold_case=not source.cased)


# ---------------------------------------------------------------------------
# GloVe
# ---------------------------------------------------------------------------


class GloveMatrix:
    """The GloVe vectors, memory-mapped from the .npz built by fetch_glove.py.

    Loaded once and held for the life of the process. 50k words at 300
    dimensions in float32 is 60 MB, which is worth it to avoid re-reading a
    1 GB text file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._words: list[str] | None = None
        self._index: dict[str, int] | None = None
        self._vectors: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        if self._vectors is not None:
            return
        if not self.available:
            raise EmbeddingError(
                f"GloVe vectors not found at {self.path}. Build them once with "
                "`python scripts/fetch_glove.py` (downloads ~820 MB, keeps 60 MB)."
            )
        logger.info(f"Loading GloVe matrix from {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            self._vectors = data["vectors"].astype(np.float32)
            self._words = [str(w) for w in data["words"]]
        self._index = {w: i for i, w in enumerate(self._words)}
        logger.info(
            f"✓ GloVe loaded: {len(self._words)} words x {self._vectors.shape[1]} dims"
        )

    @property
    def vocab_size(self) -> int:
        if not self.available:
            return 0
        self.load()
        assert self._words is not None
        return len(self._words)

    def get(self, word: str) -> np.ndarray | None:
        self.load()
        assert self._index is not None and self._vectors is not None
        idx = self._index.get(word)
        if idx is None:
            return None
        return self._vectors[idx]

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        """Vectors for the texts it knows, plus the list of the ones it does not.

        Multi-word input is averaged over its tokens. That is a real baseline
        and a poor demo, so `Source.supports_phrases` is False for GloVe and
        the UI says so — but averaging beats erroring on "new york".
        """
        self.load()
        assert self._vectors is not None
        vectors: list[np.ndarray] = []
        missing: list[str] = []
        for text in texts:
            tokens = text.split(" ")
            found = [v for v in (self.get(t) for t in tokens) if v is not None]
            if not found:
                missing.append(text)
                continue
            vectors.append(np.mean(found, axis=0) if len(found) > 1 else found[0])
        if not vectors:
            return np.zeros((0, self.dims), dtype=np.float32), missing
        return np.vstack(vectors).astype(np.float32), missing

    @property
    def dims(self) -> int:
        self.load()
        assert self._vectors is not None
        return int(self._vectors.shape[1])

    def nearest(self, vector: np.ndarray, topk: int, exclude: set[str]) -> list[tuple[str, float]]:
        """Nearest words in the *whole* GloVe vocabulary, not just the index.

        This is what makes analogies satisfying. Searching only the words the
        user happens to have added means king − man + woman can only return
        something already on screen; searching 50k words means it can return
        queen because queen is genuinely the closest point in the space.
        """
        self.load()
        assert self._vectors is not None and self._words is not None
        norms = np.linalg.norm(self._vectors, axis=1)
        norms[norms == 0] = 1.0
        query = vector / (np.linalg.norm(vector) or 1.0)
        sims = (self._vectors @ query) / norms
        # Take extra so there is still a full list left after the inputs are
        # dropped, then sort only that slice.
        want = min(len(sims), topk + len(exclude) + 1)
        candidates = np.argpartition(-sims, want - 1)[:want]
        candidates = candidates[np.argsort(-sims[candidates])]
        out: list[tuple[str, float]] = []
        for idx in candidates:
            word = self._words[idx]
            if word in exclude:
                continue
            out.append((word, float(sims[idx])))
            if len(out) == topk:
                break
        return out


glove = GloveMatrix(settings.glove_path)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


async def embed_openai(model: str, texts: list[str]) -> np.ndarray:
    if not settings.openai_configured:
        raise EmbeddingError(
            "No OpenAI key configured. Set OPENAI_KEY in .env, or switch the "
            "source to GloVe, which runs entirely locally."
        )
    vectors: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {settings.openai_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        for start in range(0, len(texts), settings.embed_batch_size):
            batch = texts[start : start + settings.embed_batch_size]
            try:
                response = await client.post(
                    f"{settings.openai_base_url}/embeddings",
                    headers=headers,
                    json={"model": model, "input": batch},
                )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Could not reach the OpenAI API: {exc}") from exc
            if response.status_code != 200:
                # The API's own message is more useful than anything invented
                # here — it names the model, the quota or the bad key.
                detail = response.text[:300]
                raise EmbeddingError(
                    f"OpenAI returned {response.status_code}: {detail}"
                )
            payload = response.json()
            # The API documents that data comes back in input order, but it
            # also returns an index on every item. Sorting by it is free.
            items = sorted(payload["data"], key=lambda d: d["index"])
            vectors.extend(item["embedding"] for item in items)
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


async def embed_cohere(model: str, input_type: str, texts: list[str]) -> np.ndarray:
    """Vectors from Cohere's /v2/embed, in the order the texts were given.

    Deliberately the same shape as `embed_openai` — one httpx POST per batch,
    the API's own error text passed through — with three differences that are
    the API's, not ours:

      * `input_type` is required, and it changes the vector rather than
        annotating it. `Source.input_type` says why this one is `clustering`.
      * 96 texts per call is a hard maximum, enforced in settings.py, so this
        loop cannot be widened by raising EMBED_BATCH_SIZE.
      * the rows come back with no index on them, so they cannot be re-sorted
        into the order they were sent. They are documented as already being in
        it; the length check below is what makes that assumption fail loudly
        instead of filing every vector under the wrong word.
    """
    if not settings.cohere_configured:
        raise EmbeddingError(
            "No Cohere key configured. Set COHERE_KEY in .env, or switch the "
            "source to GloVe, which runs entirely locally."
        )
    vectors: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {settings.cohere_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        for start in range(0, len(texts), settings.cohere_batch_size):
            batch = texts[start : start + settings.cohere_batch_size]
            try:
                response = await client.post(
                    f"{settings.cohere_base_url}/embed",
                    headers=headers,
                    json={
                        "model": model,
                        "texts": batch,
                        "input_type": input_type,
                        # float is the documented default, and asking for it
                        # explicitly costs a line: the other options are int8
                        # and binary, which would deserialise perfectly and be
                        # quantised vectors nothing here would notice.
                        "embedding_types": ["float"],
                        # Words and short sentences, so nothing should truncate.
                        # END rather than NONE because a 400 on one long input
                        # would lose the whole batch with it.
                        "truncate": "END",
                    },
                )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Could not reach the Cohere API: {exc}") from exc
            if response.status_code != 200:
                detail = response.text[:300]
                raise EmbeddingError(
                    f"Cohere returned {response.status_code}: {detail}"
                )
            rows = (response.json().get("embeddings") or {}).get("float")
            if not isinstance(rows, list):
                raise EmbeddingError(
                    "Cohere returned no float embeddings — asked for "
                    f"embedding_types=['float'] from {model}."
                )
            if len(rows) != len(batch):
                raise EmbeddingError(
                    f"Cohere returned {len(rows)} vectors for {len(batch)} "
                    "texts; the rows carry no index, so they cannot be "
                    "realigned with the words that produced them."
                )
            vectors.extend(rows)
    return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


async def embed(source: Source, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Vectors for `texts` under `source`, plus any texts it could not embed.

    Returns raw vectors, not normalised ones. Cosine similarity normalises at
    comparison time anyway, and the arithmetic in app/services/similarity.py
    reads better on the vectors the model actually produced.
    """
    if not texts:
        return np.zeros((0, source.dims), dtype=np.float32), []

    if source.pivot:
        # The pivot index: translate, then embed the translation with the same
        # model. Duplicates are passed through rather than collapsed — when
        # "eau" and "water" both become "water" they genuinely do get the same
        # vector, and hiding that would hide the pivot's defining property.
        from app.services import translate

        translations = await translate.to_english(texts)
        english = [
            translations[t].english if t in translations else t for t in texts
        ]
        vectors, missing_english = await embed(base_of(source), english)
        # A word is missing from the pivot index when its *translation* was
        # missing, so the report has to be mapped back to what the caller asked
        # for. Order is preserved on both sides, which is what lets the rows of
        # `vectors` still line up with the texts that survived.
        unusable = set(missing_english)
        missing = [t for t, e in zip(texts, english) if e in unusable]
        return vectors, missing

    if source.provider == "glove":
        return glove.embed(texts)
    if source.provider == "openai":
        return await embed_openai(source.model, texts), []
    if source.provider == "cohere":
        return await embed_cohere(source.model, source.input_type, texts), []
    if source.provider == "onnx":
        # A forward pass is CPU-bound and holds the event loop, so it goes to
        # a worker thread. Without this, embedding forty words through LaBSE
        # would stall every other request in the process for a second.
        encoder = encoder_for(source.id)
        try:
            return await asyncio.to_thread(encoder.embed, texts), []
        except FileNotFoundError as exc:
            raise EmbeddingError(str(exc)) from exc
    raise EmbeddingError(f"Unknown provider {source.provider!r}")


def source_status(source: Source) -> tuple[bool, str]:
    """Whether a source can serve requests right now, and why not if it cannot."""
    if source.pivot:
        # A pivot source is exactly as available as the model behind it. The
        # translator is not part of that test: the built-in word sets carry
        # their own exact translations, so the pivot index works with no key
        # at all until someone types a word that is not in them.
        return source_status(base_of(source))
    if source.provider == "openai":
        if not settings.openai_configured:
            return False, "OPENAI_KEY is not set"
        return True, ""
    if source.provider == "cohere":
        if not settings.cohere_configured:
            return False, "COHERE_KEY is not set"
        return True, ""
    if source.provider == "glove":
        if not glove.available:
            return False, "run scripts/fetch_glove.py to build the vector file"
        return True, ""
    if source.provider == "onnx":
        if not encoder_for(source.id).available:
            return False, (
                f"run scripts/fetch_models.py --model {source.id} to download "
                "the weights"
            )
        return True, ""
    return False, "unknown provider"
