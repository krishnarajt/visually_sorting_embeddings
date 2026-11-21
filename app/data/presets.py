"""The words the UI offers, and why each set is worth looking at.

A blank 3-D scatter plot teaches nothing — the whole value of this thing is in
which words you put in it. Every set below was chosen to make one specific
property of embedding space visible, and `blurb` is what the UI prints next to
it so the user knows what to look for before they click.

`PRESETS` fills the word box. `ANALOGIES` fills the arithmetic bar.
`SAMPLER` is the loose pile of interesting single words offered as chips.
"""

from dataclasses import dataclass

from app.data.multilingual import LEXICONS


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    blurb: str
    words: list[str]
    # Set when the words are sentences rather than single tokens. GloVe has no
    # vector for a sentence — it has vectors for the words in it, and averaging
    # them is a real technique that makes a dishonest demo. Rather than list
    # the models that qualify, which was a list that went stale the moment
    # three more sentence encoders arrived, this asks for the capability and
    # /api/meta resolves it against Source.supports_phrases.
    requires_phrases: bool = False
    # What to actually do once it is loaded. Shown as a hint in the UI.
    look_for: str = ""
    # Set when the words came from a concept-by-language grid in
    # app/data/multilingual.py. Those words carry a concept and a language
    # each, which is what lets /api/alignment put a number on the set and the
    # UI colour by either axis. Empty for the plain word lists.
    lexicon: str = ""


PRESETS: list[Preset] = [
    Preset(
        id="royalty",
        label="Royalty and gender",
        blurb=(
            "The classic. Pairs that differ only in gender, plus the titles that go "
            "with them."
        ),
        look_for=(
            "The pairs should sit parallel: the arrow from man to woman points the "
            "same way as the one from king to queen. That parallelism is what makes "
            "king − man + woman work at all."
        ),
        words=[
            "king", "queen", "man", "woman", "prince", "princess", "boy", "girl",
            "father", "mother", "son", "daughter", "uncle", "aunt", "nephew",
            "niece", "husband", "wife", "brother", "sister", "actor", "actress",
            "waiter", "waitress", "emperor", "empress",
        ],
    ),
    Preset(
        id="capitals",
        label="Countries and capitals",
        blurb="Nine countries next to the cities that run them.",
        look_for=(
            "Countries cluster with countries and cities with cities, but the "
            "country-to-capital offset is roughly constant — so paris − france + "
            "japan lands near tokyo."
        ),
        words=[
            "france", "paris", "japan", "tokyo", "italy", "rome", "germany",
            "berlin", "spain", "madrid", "egypt", "cairo", "india", "delhi",
            "canada", "ottawa", "brazil", "russia", "moscow", "china", "beijing",
        ],
    ),
    Preset(
        id="clusters",
        label="Six unrelated fields",
        blurb=(
            "Fruit, vehicles, instruments, emotions, weather and colours, shuffled "
            "together."
        ),
        look_for=(
            "The one set where every method agrees. Six clumps should fall out "
            "without being told there are six — the structure is in the vectors, "
            "not in the colouring."
        ),
        words=[
            "apple", "banana", "mango", "grape", "peach", "cherry",
            "car", "truck", "bicycle", "train", "airplane", "boat",
            "piano", "guitar", "violin", "drums", "flute", "trumpet",
            "happy", "sad", "angry", "afraid", "jealous", "calm",
            "rain", "snow", "thunder", "fog", "sunshine", "hail",
            "red", "blue", "green", "yellow", "purple", "orange",
        ],
    ),
    Preset(
        id="polysemy",
        label="Words with two meanings",
        blurb=(
            "bank, bat, apple, python, mercury — each with the company both of its "
            "senses keep."
        ),
        look_for=(
            "A word has exactly one vector, so a word with two meanings gets "
            "stranded between them. Watch 'python' sit halfway between the snakes "
            "and the programming, belonging to neither."
        ),
        words=[
            "bank", "river", "money", "loan", "shore",
            "bat", "cave", "baseball", "wings", "pitcher",
            "apple", "fruit", "orchard", "computer", "iphone",
            "python", "snake", "programming", "code", "reptile",
            "mercury", "planet", "metal", "thermometer", "roman",
        ],
    ),
    Preset(
        id="morphology",
        label="Tense and inflection",
        blurb="The same verbs in every form they take.",
        look_for=(
            "Grammar has a geometry. The step from walk to walking is close to the "
            "step from swim to swimming, which is why the arithmetic works on "
            "conjugation as well as on meaning."
        ),
        words=[
            "walk", "walked", "walking", "run", "ran", "running",
            "swim", "swam", "swimming", "eat", "ate", "eaten",
            "go", "went", "gone", "write", "wrote", "written",
            "sing", "sang", "sung", "take", "took", "taken",
        ],
    ),
    Preset(
        id="comparatives",
        label="Good, better, best",
        blurb="Adjectives with their comparative and superlative forms.",
        look_for=(
            "Three loose shelves — base, -er, -est — with each word's own meaning "
            "deciding where it sits along the shelf. Intensity and grammar are two "
            "separate directions in the same space."
        ),
        words=[
            "good", "better", "best", "bad", "worse", "worst",
            "big", "bigger", "biggest", "small", "smaller", "smallest",
            "fast", "faster", "fastest", "slow", "slower", "slowest",
            "strong", "stronger", "strongest", "weak", "weaker", "weakest",
        ],
    ),
    Preset(
        id="opposites",
        label="Opposites",
        blurb="Antonym pairs, which embeddings are famously bad at separating.",
        look_for=(
            "hot and cold come out as near neighbours. Nothing is broken: they are "
            "used in the same sentences about the same things, and these models "
            "encode context, not truth. Opposites are a known blind spot."
        ),
        words=[
            "hot", "cold", "up", "down", "left", "right", "day", "night",
            "always", "never", "love", "hate", "war", "peace", "buy", "sell",
            "open", "closed", "true", "false", "rich", "poor",
        ],
    ),
    Preset(
        id="sentences",
        label="Sentences, not words",
        blurb=(
            "Paraphrase pairs and one intruder. Needs a model that embeds "
            "sentences."
        ),
        requires_phrases=True,
        look_for=(
            "This is what the OpenAI models were actually built for. Each pair says "
            "the same thing in different words and should land almost on top of "
            "itself, while the topic changes sit far apart. GloVe cannot do this at "
            "all — it has no vector for a sentence."
        ),
        words=[
            "the cat sat on the mat",
            "a feline rested on a rug",
            "the dog barked at the postman",
            "a hound howled at the mail carrier",
            "stock markets fell sharply today",
            "equities dropped steeply this session",
            "preheat the oven to 200 degrees",
            "set the oven at 200C before baking",
            "my flight has been delayed again",
            "the plane is running late once more",
        ],
    ),
    Preset(
        id="homographs",
        label="One string, two languages",
        blurb=(
            "Eight strings that are ordinary words in both English and German, "
            "meaning unrelated things, each with the English word for both of "
            "its senses."
        ),
        look_for=(
            "The limit of the whole idea. A model is handed a string, not a "
            "language, so 'gift' gets exactly one vector to cover an English "
            "present and a German poison. Check each trio: is the string "
            "nearer its English sense or its German one? English wins seven of "
            "eight on GloVe, on OpenAI 3-small and on minilm-multilingual, and "
            "eight of eight on LaBSE — so the exceptions are the interesting "
            "part. The word models both read 'kind' as the German for child "
            "rather than the English for caring, and minilm-multilingual reads "
            "'gut' as the German for good at 0.83 against 0.23 for intestine. "
            "The more languages a model knows, the more ways there are for a "
            "bare string to be ambiguous, and nothing in the input can resolve "
            "it."
        ),
        words=[
            "gift", "present", "poison",
            "brand", "trademark", "fire",
            "rat", "rodent", "advice",
            "hell", "underworld", "bright",
            "bad", "awful", "bath",
            "kind", "caring", "child",
            "gut", "intestine", "good",
            "handy", "useful", "phone",
        ],
    ),
    Preset(
        id="numbers",
        label="Numbers and time",
        blurb="Digits, number words, days and months.",
        look_for=(
            "Ordered things often come out as an arc or a line rather than a blob. "
            "Look for the days of the week in order — the model was never told they "
            "have an order."
        ),
        words=[
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday", "january", "february", "march",
            "june", "july", "december",
        ],
    ),
]


@dataclass(frozen=True)
class Analogy:
    label: str
    # Signed terms: +1 adds the vector, -1 subtracts it.
    terms: list[tuple[str, int]]
    expect: str
    note: str = ""


ANALOGIES: list[Analogy] = [
    Analogy(
        label="king − man + woman",
        terms=[("king", 1), ("man", -1), ("woman", 1)],
        expect="queen",
        note="The one everybody has read about. Works cleanly on GloVe.",
    ),
    Analogy(
        label="paris − france + japan",
        terms=[("paris", 1), ("france", -1), ("japan", 1)],
        expect="tokyo",
        note="Geography, held in the same kind of offset as gender.",
    ),
    Analogy(
        label="uncle − man + woman",
        terms=[("uncle", 1), ("man", -1), ("woman", 1)],
        expect="aunt",
        note="Kinship. Load the royalty preset first so aunt is in the index.",
    ),
    Analogy(
        label="walking − walk + swim",
        terms=[("walking", 1), ("walk", -1), ("swim", 1)],
        expect="swimming",
        note="Grammar rather than meaning — the -ing direction.",
    ),
    Analogy(
        label="bigger − big + small",
        terms=[("bigger", 1), ("big", -1), ("small", 1)],
        expect="smaller",
        note=(
            "A real near-miss on GloVe: it answers 'larger' by 0.002, with "
            "'smaller' second. The comparative direction is there, the "
            "antonym is what slips."
        ),
    ),
    Analogy(
        label="best − good + bad",
        terms=[("best", 1), ("good", -1), ("bad", 1)],
        expect="worst",
        note="Superlatives. Shakier than the others; a good failure to look at.",
    ),
]


# Single words worth trying, offered as clickable chips. Deliberately a mixed
# bag: concrete nouns to anchor the space, then abstractions, proper nouns and
# a few words that behave oddly.
SAMPLER: list[str] = [
    "dog", "cat", "wolf", "kitten", "puppy", "lion", "elephant", "shark",
    "doctor", "nurse", "teacher", "engineer", "lawyer", "farmer",
    "coffee", "tea", "bread", "cheese", "pizza", "soup",
    "london", "tokyo", "cairo", "mumbai", "toronto",
    "guitar", "piano", "cello",
    "justice", "freedom", "hope", "grief", "courage", "boredom",
    "einstein", "shakespeare", "mozart", "darwin",
    "kubernetes", "docker", "python", "javascript",
    "ocean", "mountain", "desert", "forest", "glacier",
    "war", "peace", "money", "debt", "tax",
]


# The cross-lingual grids, appended as ordinary word sets so that every
# control in the UI keeps working on them without knowing they are special.
# What makes them special is `lexicon`, which points back at the grid the
# labels live in.
PRESETS += [
    Preset(
        id=lexicon.id,
        label=lexicon.label,
        blurb=lexicon.blurb,
        look_for=lexicon.look_for,
        words=lexicon.words(),
        requires_phrases=lexicon.requires_phrases,
        lexicon=lexicon.id,
    )
    for lexicon in LEXICONS
]


def preset_by_id(preset_id: str) -> Preset | None:
    return next((p for p in PRESETS if p.id == preset_id), None)


def all_preset_words() -> list[str]:
    """Every word in every preset, deduplicated, order preserved.

    Used by scripts/export_demo.py to decide what the offline bundle has to
    contain, and by the seed route so the analogy search has a vocabulary to
    hit before the user has typed anything.
    """
    seen: dict[str, None] = {}
    for preset in PRESETS:
        for word in preset.words:
            seen.setdefault(word, None)
    for word in SAMPLER:
        seen.setdefault(word, None)
    for analogy in ANALOGIES:
        for term, _ in analogy.terms:
            seen.setdefault(term, None)
        seen.setdefault(analogy.expect, None)
    return list(seen)
