"""Word sets built to answer one question: does the space store meaning or language?

The rest of this app asks what a projection costs. These sets ask something
about the models themselves, and they need a different shape of data to do it.
A flat list of words cannot express "these six strings are the same concept and
these eight are the same language" — so a `Lexicon` is a concept-by-language
grid, and every word in it carries both labels. `app/services/alignment.py`
reads those labels and turns them into numbers; the UI reads them and colours
by one axis or the other, which is the whole visual argument in two clicks.

## What the naive expectation is, and why it is wrong

"Words that mean the same thing are near each other" is a claim people make
about embeddings generally, and it is only true within one language. A
monolingual model has no reason to put `agua` near `water`: it saw `agua` a few
hundred times in English sentences *about Spanish*, so it learned a vector that
means "a Spanish-looking word", and the nearest thing to it is `señor`.
Language is a much stronger signal in text than meaning is, and a model will
learn whatever the data makes cheap.

Making a model cross-lingual therefore takes deliberate work, and the models
here span three ways of not doing it, doing it by accident, and doing it on
purpose. `Source.cross_lingual` in app/services/embedding.py records which.

## The grids

  translations   Eight everyday concepts in six languages. The main event.
  scripts        Five concepts in eight writing systems, no two alike. The
                 hard version: script is the most visible surface feature
                 there is, and a tokeniser sees nothing else.
  parallel       Four sentences in six languages. What the sentence encoders
                 were actually built for, and where the word models cannot
                 follow.

A fourth set, `homographs`, lives in presets.py because it has no grid: it is
strings that are words in two languages at once, which is the failure mode
these models cannot design their way out of. One string gets one vector.

## On `pain`

French for bread is spelled exactly like English for suffering, and it is left
in the translations grid on purpose: it is the one cell where form and meaning
give opposite answers, so it is the one cell worth looking at first.

It also turned out to be the single most discriminating point in the set, and
not along the axis you would expect. Asked for the nearest word in another
language, the two OpenAI models and multilingual-e5-small all answer `pan` and
LaBSE answers `хлеб` — the concept wins. GloVe answers `mother`, having learned
nothing about the string except that it is a word. And
minilm-multilingual — which takes the best separation scores on every grid in
this file — answers `sol`, the Spanish for sun, with `pan` only third. A model
can be the best in the room at holding translations together and still be the
one that trips over a homograph. That is why the confusion list in
app/services/alignment.py is reported alongside the percentage: 94% with `pain`
in the misses is a different model from 94% without it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    label: str
    # What speakers call it. Shown in the legend, because a set about languages
    # that names them all in English is making a small version of the mistake
    # it is trying to measure.
    endonym: str
    script: str


@dataclass(frozen=True)
class Concept:
    id: str
    # The English word, used as the concept's name in the UI. That English is
    # the metalanguage here is a convenience, not a claim about the geometry.
    gloss: str
    # language code -> the word in that language
    words: dict[str, str]


@dataclass(frozen=True)
class Lexicon:
    """A concept-by-language grid of words, and what to look for in it."""

    id: str
    label: str
    blurb: str
    look_for: str
    languages: list[Language]
    concepts: list[Concept]
    # Set where the words are sentences: GloVe has no vector for one, and
    # averaging its tokens is a real technique that makes a dishonest demo.
    # Resolved against Source.supports_phrases in /api/meta.
    requires_phrases: bool = False

    def words(self) -> list[str]:
        """Every word, grouped by concept.

        Concept-major rather than language-major so that a word and its
        translations are adjacent in the list. Nothing downstream depends on
        the order — but the word chips in the UI are rendered in it, and
        reading the grid row by row is how you check a translation.
        """
        return [
            concept.words[language.code]
            for concept in self.concepts
            for language in self.languages
            if language.code in concept.words
        ]

    def entries(self) -> list[tuple[str, str, str]]:
        """(word, language code, concept id) for every cell in the grid."""
        return [
            (concept.words[language.code], language.code, concept.id)
            for concept in self.concepts
            for language in self.languages
            if language.code in concept.words
        ]

    @property
    def language_codes(self) -> list[str]:
        return [language.code for language in self.languages]


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

EN = Language("en", "English", "English", "Latin")
FR = Language("fr", "French", "français", "Latin")
ES = Language("es", "Spanish", "español", "Latin")
DE = Language("de", "German", "Deutsch", "Latin")
RU = Language("ru", "Russian", "русский", "Cyrillic")
HI = Language("hi", "Hindi", "हिन्दी", "Devanagari")
EL = Language("el", "Greek", "ελληνικά", "Greek")
HE = Language("he", "Hebrew", "עברית", "Hebrew")
AR = Language("ar", "Arabic", "العربية", "Arabic")
JA = Language("ja", "Japanese", "日本語", "Japanese")
KO = Language("ko", "Korean", "한국어", "Hangul")


# ---------------------------------------------------------------------------
# The grids
# ---------------------------------------------------------------------------

LEXICONS: list[Lexicon] = [
    Lexicon(
        id="translations",
        label="The same word in six languages",
        blurb=(
            "Eight everyday concepts — water, dog, house, book, mother, bread, "
            "sun, moon — in English, French, Spanish, German, Russian and Hindi."
        ),
        look_for=(
            "Colour by language, then by meaning, and see which one the plot was "
            "already sorted by. On GloVe the languages separate and the concepts "
            "do not; on LaBSE the eight concepts fall into eight groups of six "
            "and the languages are nowhere. Then find the French 'pain' — "
            "spelled like the English word for suffering, and the one point "
            "where form and meaning disagree. LaBSE puts it with 'хлеб'; GloVe "
            "puts it with 'mother'; and minilm-multilingual, which otherwise "
            "wins this set, puts it with 'sol'."
        ),
        languages=[EN, FR, ES, DE, RU, HI],
        concepts=[
            Concept("water", "water", {
                "en": "water", "fr": "eau", "es": "agua",
                "de": "Wasser", "ru": "вода", "hi": "पानी",
            }),
            Concept("dog", "dog", {
                "en": "dog", "fr": "chien", "es": "perro",
                "de": "Hund", "ru": "собака", "hi": "कुत्ता",
            }),
            Concept("house", "house", {
                "en": "house", "fr": "maison", "es": "casa",
                "de": "Haus", "ru": "дом", "hi": "घर",
            }),
            Concept("book", "book", {
                "en": "book", "fr": "livre", "es": "libro",
                "de": "Buch", "ru": "книга", "hi": "किताब",
            }),
            Concept("mother", "mother", {
                "en": "mother", "fr": "mère", "es": "madre",
                "de": "Mutter", "ru": "мать", "hi": "माँ",
            }),
            Concept("bread", "bread", {
                "en": "bread", "fr": "pain", "es": "pan",
                "de": "Brot", "ru": "хлеб", "hi": "रोटी",
            }),
            Concept("sun", "sun", {
                "en": "sun", "fr": "soleil", "es": "sol",
                "de": "Sonne", "ru": "солнце", "hi": "सूरज",
            }),
            Concept("moon", "moon", {
                "en": "moon", "fr": "lune", "es": "luna",
                "de": "Mond", "ru": "луна", "hi": "चाँद",
            }),
        ],
    ),
    Lexicon(
        id="scripts",
        label="Eight writing systems",
        blurb=(
            "Five concepts, eight languages, and no two of them written in the "
            "same alphabet."
        ),
        look_for=(
            "The hardest version of the question. A tokeniser sees characters, "
            "and nothing about 水 and 물 and وشمس looks alike at that level — so "
            "any model that groups these by meaning is doing it despite the "
            "surface, not because of it. This is also where GloVe stops being "
            "wrong and starts being absent: it has no vector for a single "
            "non-Latin character string at all."
        ),
        languages=[EN, EL, RU, HE, AR, HI, JA, KO],
        concepts=[
            Concept("water", "water", {
                "en": "water", "el": "νερό", "ru": "вода", "he": "מים",
                "ar": "ماء", "hi": "पानी", "ja": "水", "ko": "물",
            }),
            Concept("mother", "mother", {
                "en": "mother", "el": "μητέρα", "ru": "мать", "he": "אמא",
                "ar": "أم", "hi": "माँ", "ja": "母", "ko": "어머니",
            }),
            Concept("sun", "sun", {
                "en": "sun", "el": "ήλιος", "ru": "солнце", "he": "שמש",
                "ar": "شمس", "hi": "सूरज", "ja": "太陽", "ko": "태양",
            }),
            Concept("book", "book", {
                "en": "book", "el": "βιβλίο", "ru": "книга", "he": "ספר",
                "ar": "كتاب", "hi": "किताब", "ja": "本", "ko": "책",
            }),
            Concept("fire", "fire", {
                "en": "fire", "el": "φωτιά", "ru": "огонь", "he": "אש",
                "ar": "نار", "hi": "आग", "ja": "火", "ko": "불",
            }),
        ],
    ),
    Lexicon(
        id="parallel",
        label="The same sentence in six languages",
        blurb=(
            "Four sentences, each translated five times. Needs a model that "
            "embeds sentences."
        ),
        requires_phrases=True,
        look_for=(
            "Sentences are what these encoders were trained on, so this is the "
            "kindest test in the app and the scores should be the highest you "
            "see. A sentence gives the model far more to work with than a bare "
            "noun does — there is grammar, tense and topic in it, and all of "
            "that survives translation. If a model is going to make language "
            "invisible anywhere, it is here."
        ),
        languages=[EN, FR, ES, DE, RU, HI],
        concepts=[
            Concept("rain", "it is raining heavily today", {
                "en": "it is raining heavily today",
                "fr": "il pleut beaucoup aujourd'hui",
                "es": "hoy llueve mucho",
                "de": "heute regnet es stark",
                "ru": "сегодня сильный дождь",
                "hi": "आज बहुत बारिश हो रही है",
            }),
            Concept("train", "the train leaves at nine in the morning", {
                "en": "the train leaves at nine in the morning",
                "fr": "le train part à neuf heures du matin",
                "es": "el tren sale a las nueve de la mañana",
                "de": "der Zug fährt um neun Uhr morgens ab",
                "ru": "поезд отходит в девять часов утра",
                "hi": "ट्रेन सुबह नौ बजे निकलती है",
            }),
            Concept("reading", "she loves reading books in the evening", {
                "en": "she loves reading books in the evening",
                "fr": "elle aime lire des livres le soir",
                "es": "le encanta leer libros por la tarde",
                "de": "sie liest abends gern Bücher",
                "ru": "она любит читать книги вечером",
                "hi": "उसे शाम को किताबें पढ़ना पसंद है",
            }),
            Concept("price", "how much does this cost", {
                "en": "how much does this cost",
                "fr": "combien est-ce que ça coûte",
                "es": "cuánto cuesta esto",
                "de": "wie viel kostet das",
                "ru": "сколько это стоит",
                "hi": "यह कितने का है",
            }),
        ],
    ),
]

LEXICONS_BY_ID: dict[str, Lexicon] = {lexicon.id: lexicon for lexicon in LEXICONS}


def lexicon_by_id(lexicon_id: str) -> Lexicon | None:
    return LEXICONS_BY_ID.get(lexicon_id)


def all_languages() -> list[Language]:
    """Every language used by any grid, deduplicated, for the UI legend."""
    seen: dict[str, Language] = {}
    for lexicon in LEXICONS:
        for language in lexicon.languages:
            seen.setdefault(language.code, language)
    return list(seen.values())
