"""Translating text to English, so a monolingual index can be compared fairly.

There are two ways to build a search index over documents in many languages.
Embed each document in the language it was written in and rely on the model to
place translations together — or translate everything to English first,
translate the query to English too, and run an ordinary English index. The
second is the pivot, or translate-to-English, approach, and it is how most
multilingual search actually worked before cross-lingual encoders were good.

The comparison is not obvious in either direction:

  native  keeps everything the model knows about the original wording, and
          costs one embedding call. It is only as cross-lingual as the model
          is, which for GloVe is not at all.
  pivot   throws the original wording away and inherits the translator's
          mistakes, but afterwards every vector is an English vector, so
          *any* English model — including a 2014 word model with no
          multilingual training whatsoever — retrieves across languages.

app/services/alignment.py measures both, and `/api/pivot` runs them side by
side. The result is not the one you would guess from the model cards.

## Two translators, and why the exact one matters

`GLOSSARY` is built from the word sets in app/data/multilingual.py, which
already carry the English word for every concept — that is what makes them a
grid. So for anything built in, the translation is exact, free, offline and
identical on every run.

That is deliberately the *upper bound* on the pivot approach. With a perfect
translator, a pivot index is as good as a pivot index can ever be, and the
comparison isolates the one question worth asking: is translating first better
than a cross-lingual model, when translation is free and correct? Any real
deployment does worse, because a real deployment has the second translator.

The OpenAI translator handles anything the glossary does not, which means any
word a user types. It is cached to disk, because a teaching tool that bills
per page reload is a bad teaching tool, and because two runs of
scripts/export_demo.py should produce the same bundle.

## The `pain` wart

A string is attested as English somewhere in the word sets, or it is not.
`pain` is not — the only lexicon that contains it is the French for bread — so
the glossary translates `pain` to `bread`, and a user who typed `pain` meaning
suffering gets a surprise. Resolving that properly needs the language of the
input, which is exactly the information a text index does not have; it is the
same ambiguity the `homographs` word set is about, arriving from the other
direction. English entries do win over other languages for the same string, so
this only bites on strings that are English by accident.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass

import httpx

from app.data.multilingual import LEXICONS
from app.settings import settings

logger = logging.getLogger(__name__)


def _build_glossary() -> dict[str, str]:
    """Every *word* in every grid, mapped to that concept's English word.

    Sentence grids are deliberately excluded. A dictionary translates words;
    it does not translate sentences, and pretending otherwise would break the
    comparison this module exists for. The grid's English sentence is sitting
    right there next to its translations, so a glossary built from it would
    map all six of them to a byte-identical string — the pivot index would
    score a perfect 1.000 because every concept had collapsed to one point,
    and it would have learned nothing except that we handed it the answer.
    Sentences therefore go through the real translator, whose output differs
    from the grid's English in exactly the way a real translation does.

    Built in two passes so that a string attested as English translates to
    itself no matter what other language also spells a word that way.
    """
    glossary: dict[str, str] = {}
    word_grids = [lexicon for lexicon in LEXICONS if not lexicon.requires_phrases]
    for lexicon in word_grids:
        for concept in lexicon.concepts:
            english = concept.words.get("en")
            if not english:
                continue
            for code, word in concept.words.items():
                if code != "en":
                    glossary[word.strip().lower()] = english
    for lexicon in word_grids:
        for concept in lexicon.concepts:
            english = concept.words.get("en")
            if english:
                glossary[english.strip().lower()] = english
    return glossary


GLOSSARY: dict[str, str] = _build_glossary()


class TranslationError(RuntimeError):
    """Raised when text cannot be translated, with a message for the UI."""


@dataclass(frozen=True)
class Translation:
    source_text: str
    english: str
    # "glossary" — exact, from the word sets. "model" — the OpenAI translator.
    # "cache" — the model, on an earlier run. Surfaced all the way to the UI
    # because a pivot index built by a machine translator contains that
    # translator's errors, and which words those are is not guessable.
    via: str


# ---------------------------------------------------------------------------
# The disk cache
# ---------------------------------------------------------------------------

_cache: dict[str, str] | None = None


def _cache_path():
    return settings.data_dir / "translations.json"


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        path = _cache_path()
        try:
            _cache = json.loads(path.read_text(encoding="utf-8"))
            logger.info(f"✓ {len(_cache)} cached translations from {path.name}")
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    cache = _load_cache()
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a sibling and renamed: a half-written cache that still parses
    # is worse than no cache, because it would be believed.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=0, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# The model translator
# ---------------------------------------------------------------------------

_PROMPT = (
    "Translate each numbered line into English. Many lines are single words "
    "with no context; translate the most common everyday sense. If a line is "
    "already English, repeat it unchanged. Reply with the same number of "
    "lines, each as `<number>. <translation>` and nothing else — no notes, no "
    "alternatives, no quotation marks."
)


async def _translate_openai(texts: list[str]) -> dict[str, str]:
    if not settings.openai_configured:
        raise TranslationError(
            "Translating text that is not in the built-in word sets needs "
            "OPENAI_KEY. The built-in sets carry their own exact translations "
            "and work without a key."
        )
    out: dict[str, str] = {}
    headers = {
        "Authorization": f"Bearer {settings.openai_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.translate_timeout) as client:
        for start in range(0, len(texts), settings.translate_batch_size):
            batch = texts[start : start + settings.translate_batch_size]
            numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
            try:
                response = await client.post(
                    f"{settings.openai_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": settings.translate_model,
                        "messages": [
                            {"role": "system", "content": _PROMPT},
                            {"role": "user", "content": numbered},
                        ],
                        # A translation that changes between runs would make
                        # two pivot indexes of the same corpus disagree.
                        "temperature": 0,
                    },
                )
            except httpx.HTTPError as exc:
                raise TranslationError(f"Could not reach the translator: {exc}") from exc
            if response.status_code != 200:
                raise TranslationError(
                    f"The translator returned {response.status_code}: "
                    f"{response.text[:300]}"
                )
            content = response.json()["choices"][0]["message"]["content"]
            parsed = _parse_numbered(content, batch)
            out.update(parsed)
    return out


def _parse_numbered(content: str, batch: list[str]) -> dict[str, str]:
    """Match `1. dog` lines back onto their inputs.

    Indexed by the number the model echoed rather than by line order, because
    a model that drops or merges a line would otherwise shift every
    translation after it onto the wrong word — silently, and in a way that
    looks like a bad embedding rather than a bad parse.
    """
    by_number: dict[int, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        number, _, rest = line.partition(".")
        if not number.strip().isdigit():
            continue
        text = rest.strip().strip('"').strip()
        if text:
            by_number[int(number.strip())] = text
    out: dict[str, str] = {}
    for i, original in enumerate(batch):
        english = by_number.get(i + 1)
        if english is None:
            logger.warning(f"translator returned no line {i + 1} for {original!r}")
            continue
        out[original] = english
    return out


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def known_english(text: str) -> str | None:
    """The English for `text` if it is already known, without calling anything.

    For callers on a synchronous path — the neighbour query needs to know what
    a word was translated to in order to exclude it from its own results, and
    by then the translation has certainly happened, because the word could not
    have been embedded without it.
    """
    key = text.strip().lower()
    if key in GLOSSARY:
        return GLOSSARY[key]
    return _load_cache().get(key)


async def to_english(texts: list[str]) -> dict[str, Translation]:
    """English for every text, from the glossary where possible.

    Order of preference: the exact glossary, then the disk cache, then the
    model. Anything the model declines to translate is passed through
    unchanged rather than dropped — an untranslated word in a pivot index is a
    real and visible result, whereas a missing point is a puzzle.
    """
    out: dict[str, Translation] = {}
    todo: list[str] = []
    cache = _load_cache()

    for text in texts:
        key = text.strip().lower()
        if not key:
            continue
        if key in GLOSSARY:
            out[text] = Translation(text, GLOSSARY[key], "glossary")
        elif key in cache:
            out[text] = Translation(text, cache[key], "cache")
        elif text not in todo:
            todo.append(text)

    if todo:
        fresh = await _translate_openai(todo)
        for text in todo:
            english = fresh.get(text, text)
            out[text] = Translation(text, english, "model")
            cache[text.strip().lower()] = english
        await asyncio.to_thread(_save_cache)

    return out
