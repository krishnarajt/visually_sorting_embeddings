"""Every term the interface uses, explained once.

The app puts a lot of numbers on screen — trustworthiness, stress, silhouette,
meaning-beats-language, explained variance — and a number whose definition the
reader has to guess at is worse than no number, because they will guess. So
each one is defined here, served through `/api/meta`, and rendered by the UI as
a tooltip on the term and as a reference list for anyone who wants to read them
all at once.

Two reasons this is server-side rather than a constants file in the frontend.
It is the same reason `/api/meta` exists at all: the definition of
trustworthiness belongs in the same repository as the code that computes
trustworthiness, and a second copy living in a React component is a second
thing that has to stay true. And the offline bundle already carries every other
piece of explanatory prose, so these cost nothing extra to ship and cannot go
stale relative to the live API.

`short` has to work in a tooltip — one or two sentences, no hedging. `detail`
is for the reference list and is allowed to say why the number is worth having
and how it misleads, which is usually the more useful half.

Where a term names something computed in a specific module, the docstring there
is the long-form source and this is the summary. They are meant to agree; if
they ever disagree, the module is right.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    id: str
    label: str
    short: str
    detail: str = ""
    topic: str = ""


TERMS: list[Term] = [
    # -- the vectors themselves ------------------------------------------
    Term(
        id="embedding",
        label="embedding",
        topic="The vectors",
        short=(
            "A list of a few hundred to a few thousand numbers standing in for "
            "a word or a sentence, learned so that similar meanings get similar "
            "lists."
        ),
        detail=(
            "Every model here turns text into one fixed-length list of numbers. "
            "The numbers are not features anyone chose — they came out of "
            "training — so no single one of them means anything on its own. "
            "What they carry is relative: the interesting question is never "
            "what a vector *is* but which other vectors it is near."
        ),
    ),
    Term(
        id="cosine",
        label="cosine similarity",
        topic="The vectors",
        short=(
            "The angle between two vectors, as a number from -1 to 1. 1 is the "
            "same direction, 0 is unrelated."
        ),
        detail=(
            "Cosine ignores how long the vectors are and compares only their "
            "direction, which is what you want when one word appeared a million "
            "times in training and another appeared a thousand. Every similarity "
            "in this app is a cosine computed on the full vectors — never on the "
            "three coordinates a projection produced. In practice fewer than 2% "
            "of real word pairs come out negative, so the colour ramp starts at "
            "0 and the theoretical floor of -1 is never reached."
        ),
    ),
    Term(
        id="dimension",
        label="dimension",
        topic="The vectors",
        short=(
            "One position in the list. Individually meaningless: there is no "
            "'water dimension'."
        ),
        detail=(
            "The raw projection method exists to demonstrate this. On LaBSE the "
            "three most varied dimensions out of 768 account for 1.1% of the "
            "variation between words, where three PCA components — learned "
            "combinations of all 768 — account for 46.1%. But the *set* of "
            "dimensions a word drives hardest is partly shared by its "
            "translations: `water` and `вода` share all five of their top five, "
            "`book` and `livre` share one. Across the translations grid the "
            "rank correlation between that overlap and cosine has a median of "
            "+0.53 — real, and loose. So meaning is spread across a combination "
            "rather than stored in any one place, and the combination is still "
            "partly legible. Both halves are measured: the first by raw mode, "
            "the second by the dimensions panel."
        ),
    ),
    Term(
        id="hub-dimension",
        label="hub dimension",
        topic="The vectors",
        short=(
            "A dimension that is large for almost every word, carrying a global "
            "offset rather than anything about the word."
        ),
        detail=(
            "These models have a handful of dimensions — sometimes called rogue "
            "or hub dimensions — with a large value for nearly every input. "
            "Dimension 216 is in the top five of `water`, `dog` and `book` on "
            "LaBSE. Ranking a word's dimensions by raw size therefore mostly "
            "returns hubs, which is the most convincing wrong answer in the "
            "area. Ranking by distance from the mean removes them for free, "
            "because a dimension that is large for everyone deviates for no "
            "one — but it removes much of the real signal with them, so the "
            "dimensions panel flags hubs instead of subtracting them, and also "
            "reports the overlap with them deleted so you can see what they "
            "were worth. How much they matter differs sharply by model: bge-m3 "
            "has five of them where multilingual-e5-small has none."
        ),
    ),
    Term(
        id="input-type",
        label="input type",
        topic="The vectors",
        short=(
            "Cohere's embed v3 makes you declare what the vector is for. The "
            "same string embedded for search and for clustering comes back as "
            "two different vectors."
        ),
        detail=(
            "Asymmetric search needs a three-word query and a three-paragraph "
            "document to land near each other, so `search_query` and "
            "`search_document` are trained to be read against each other "
            "rather than against themselves. Every number in this app is a "
            "cosine between two items drawn from the same pool — the "
            "similarity matrix, k-means, the silhouette, the translation "
            "retrieval — which is the symmetric case, so the Cohere source "
            "asks for `clustering` and never asks for anything else. Two "
            "input types in one index would be two subtly different "
            "geometries in one collection, and no error would ever be raised "
            "about it. It is the same kind of decision as the `query: ` prefix "
            "multilingual-e5 was trained with, or the pooling an ONNX export "
            "leaves to whoever runs it: not configuration, part of what the "
            "model means. The other models here take the choice away by "
            "having only one mode — which is a smaller API and a quieter "
            "assumption, not a better one."
        ),
    ),

    # -- projections ------------------------------------------------------
    Term(
        id="projection",
        label="projection",
        topic="Flattening to three",
        short=(
            "Any way of turning hundreds of dimensions into the three a screen "
            "has. All of them lose information; they differ in what they lose."
        ),
        detail=(
            "There is no faithful picture of a 768-dimensional cloud in three "
            "dimensions, so the question is never whether a projection lied but "
            "which lie it told. That is why every view here carries "
            "trustworthiness and distance correlation, and why the random "
            "projection is offered as a control: it shows what no method at all "
            "looks like, so the others can be judged against something."
        ),
    ),
    Term(
        id="trustworthiness",
        label="trustworthiness",
        topic="Flattening to three",
        short=(
            "Of each point's nearest neighbours on screen, how many were "
            "genuinely among its nearest in full dimensions. Higher is better."
        ),
        detail=(
            "This is the neighbourhood question: can you trust that two dots "
            "drawn close together really are close in the model? It says nothing "
            "about distances — t-SNE routinely scores well here while getting "
            "every distance wrong, which is exactly t-SNE's signature and the "
            "most misread property of these plots. Read it next to distance "
            "correlation, never alone."
        ),
    ),
    Term(
        id="distance-correlation",
        label="distance correlation",
        topic="Flattening to three",
        short=(
            "Rank correlation between every true cosine distance and the "
            "distance actually drawn. Higher means the picture is measurable."
        ),
        detail=(
            "Where trustworthiness asks about neighbours, this asks whether the "
            "gaps mean anything — whether a cloud twice as far away is twice as "
            "unrelated. MDS optimises this directly and scores highest; t-SNE "
            "scores around 0.37 on the six-field word set, which is why the "
            "distance between two t-SNE clusters should never be read as "
            "meaningful. A random projection scores about 0.01, which is the "
            "floor."
        ),
    ),
    Term(
        id="explained-variance",
        label="explained variance",
        topic="Flattening to three",
        short=(
            "The share of the spread between these words that the three drawn "
            "axes actually account for."
        ),
        detail=(
            "PCA reports this honestly and it is usually lower than people "
            "expect — 10 to 30% for three components on real embeddings. The raw "
            "method reports it too, alongside what three PCA components would "
            "have managed on the same words, which turns 'raw dimensions are "
            "noise' from an assertion into a ratio you can read."
        ),
    ),
    Term(
        id="stress",
        label="stress",
        topic="Flattening to three",
        short=(
            "MDS's own residual: how much disagreement is left between true and "
            "drawn distances. Lower is better."
        ),
        detail=(
            "MDS works by minimising exactly this, so stress is the one quality "
            "number here reported by the method about its own objective rather "
            "than measured from outside. It is why MDS is the only layout in the "
            "app whose axes you may sensibly measure with a ruler."
        ),
    ),
    Term(
        id="procrustes",
        label="anchoring",
        topic="Flattening to three",
        short=(
            "Rotating a new layout onto the previous one, so adding a word does "
            "not spin the whole cloud."
        ),
        detail=(
            "t-SNE and UMAP have no preferred orientation: the same data can come "
            "out mirrored or rotated on a second run, and none of that carries "
            "information. Each new frame is therefore fitted onto the last by "
            "orthogonal Procrustes, which removes the part of a change that is "
            "arbitrary and leaves the part that is real."
        ),
    ),
    Term(
        id="cluster",
        label="group",
        topic="Flattening to three",
        short=(
            "k-means run on the full vectors, not on the three coordinates — so "
            "the colouring is independent evidence, not a restatement."
        ),
        detail=(
            "Because the grouping is computed in full dimensions and the "
            "positions come from the projection, a group that also lands "
            "together on screen is two independent readings agreeing. If the "
            "colours were derived from the coordinates, the plot would be "
            "confirming itself. Colour carries the group up to three of them; "
            "past that, marker shape does, because no six hues in the lightness "
            "band a dark surface needs stay distinguishable to a colour-blind "
            "viewer."
        ),
    ),

    # -- cross-lingual ----------------------------------------------------
    Term(
        id="cross-lingual",
        label="cross-lingual",
        topic="Meaning or language",
        short=(
            "Whether a model puts a word near its translations. It takes "
            "deliberate training; it does not come free with seeing many "
            "languages."
        ),
        detail=(
            "A monolingual model has no reason to put `agua` near `water` — it "
            "saw `agua` in English sentences about Spanish, so it learned a "
            "vector meaning 'a Spanish-looking word'. Language is a stronger "
            "signal in text than meaning is. The sources here are tagged by how "
            "they came by the ability: none, incidental, multilingual (many "
            "languages, never told which strings are translations) and aligned "
            "(translation pairs as an explicit objective). The tag predicts the "
            "measurements better than parameter count does."
        ),
    ),
    Term(
        id="translation-p1",
        label="translation found",
        topic="Meaning or language",
        short=(
            "Of the words in other languages, how often the nearest one is this "
            "word's own translation."
        ),
        detail=(
            "The retrieval question, and the one a search index lives on. It is "
            "also the easiest to over-read: a model can score 100% here and "
            "still be useless for setting a similarity threshold, because "
            "ranking correctly says nothing about the margin. "
            "multilingual-e5-small scores 100% with a margin of 0.06. Always "
            "read it beside the margin, and check the list of misses — 94% with "
            "a hard case among the misses is a different model from 94% without."
        ),
    ),
    Term(
        id="margin",
        label="margin",
        topic="Meaning or language",
        short=(
            "Mean same-meaning similarity minus mean same-language similarity. "
            "How much meaning wins by, rather than whether it wins."
        ),
        detail=(
            "This is what a similarity cutoff in a real system gets set against. "
            "GloVe's is negative on the translations grid, meaning words of the "
            "same language are on average closer than translations of each "
            "other. A model with a compressed cosine range is punished here for "
            "something a user may not care about, which is why the scale-free "
            "version is reported next to it."
        ),
    ),
    Term(
        id="meaning-beats-language",
        label="meaning beats language",
        topic="Meaning or language",
        short=(
            "Every translation pair compared against every same-language pair, "
            "counting how often meaning is closer. 50% is a coin flip."
        ),
        detail=(
            "The scale-free version of the margin: it only asks which pair is "
            "closer, never by how much, so a model that squashes all its cosines "
            "into a narrow band is not penalised for it. Between this and the "
            "margin you can tell a model that ranks well and calibrates badly "
            "from one that does neither."
        ),
    ),
    Term(
        id="silhouette",
        label="silhouette",
        topic="Meaning or language",
        short=(
            "How well a labelling fits the geometry. Computed twice — by concept "
            "and by language — and the higher one is what the space is sorted by."
        ),
        detail=(
            "The only number here that can come out negative, which is the "
            "interesting case: it means real clusters exist and the label you "
            "gave them is the wrong one. GloVe scores negative by concept and "
            "positive by language, so the clusters in GloVe's space are "
            "languages. LaBSE is the other way round. Reading the two against "
            "each other is the whole question in one comparison."
        ),
    ),
    Term(
        id="cohesion",
        label="cohesion",
        topic="Meaning or language",
        short=(
            "How tightly one concept holds together across languages. The "
            "loosest concepts are the informative ones."
        ),
        detail=(
            "A concept with low cohesion is usually one whose translations are "
            "not quite synonyms, or one containing a word that is spelled like "
            "something else — which is more useful to know than the average."
        ),
    ),

    # -- translate first ---------------------------------------------------
    Term(
        id="pivot",
        label="translate first",
        topic="Translate first",
        short=(
            "Translate every document and every query to English, then run an "
            "ordinary English index. The other way to build multilingual search."
        ),
        detail=(
            "It costs a translation call per document and per query and throws "
            "the original wording away, but afterwards every vector is an "
            "English vector — so even a 2014 English-only word model retrieves "
            "across languages. Asking a translate-first GloVe index for the "
            "neighbours of `собака` returns `dogs, cat, pet, puppy, hound` out "
            "of the full 50,000-word English vocabulary. The model's language "
            "coverage stops mattering and the translator's quality starts."
        ),
    ),
    Term(
        id="collapse",
        label="collapse",
        topic="Translate first",
        short=(
            "When translation turns several documents into the same string, so "
            "the index holds one vector where the word set had several."
        ),
        detail=(
            "A dictionary turns `eau`, `agua`, `Wasser`, `вода` and `पानी` all "
            "into the literal string `water`. The model returns one vector for "
            "one string, so six documents become one and every score goes to a "
            "perfect 1.000 — by arithmetic, not by skill, because there is "
            "nothing left to retrieve across. The comparison reports this "
            "instead of the scores when it happens. It is a real finding and it "
            "cuts both ways: translation erases the language, and it erases the "
            "difference between the documents with it. Sentences are where the "
            "comparison becomes meaningful, because their translations are not "
            "byte-identical."
        ),
    ),

    # -- word sets ---------------------------------------------------------
    Term(
        id="homograph",
        label="homograph",
        topic="Word sets",
        short=(
            "One spelling that is a word in two languages with unrelated "
            "meanings. A model gets a string, not a language."
        ),
        detail=(
            "`gift` gets exactly one vector to cover an English present and a "
            "German poison, and no amount of cross-lingual training fixes that, "
            "because the information needed to resolve it was never in the "
            "input. English usually wins — it is most of the training data — so "
            "the margin is more interesting than the winner. The exceptions are "
            "the best part: two of the models read `kind` as the German for "
            "child, and the most strongly aligned one reads `gut` as the German "
            "for good."
        ),
    ),
    Term(
        id="polysemy",
        label="two meanings",
        topic="Word sets",
        short=(
            "A word with more than one sense still gets one vector, so it is "
            "stranded between them."
        ),
        detail=(
            "Watch `python` sit halfway between the snakes and the programming, "
            "belonging to neither. This is a property of any model that assigns "
            "one vector per string, and it is the same limitation the homograph "
            "set shows arriving from a different direction."
        ),
    ),
    Term(
        id="analogy",
        label="vector arithmetic",
        topic="Word sets",
        short=(
            "king − man + woman, done on the vectors, and what the result lands "
            "nearest to."
        ),
        detail=(
            "It genuinely works on GloVe — `queen` comes back at 0.69 — because "
            "GloVe is the kind of model it was discovered on. It is much weaker "
            "on a contrastively trained sentence encoder, so a miss there is a "
            "property of the model rather than a bug. The per-term similarities "
            "are shown because when an analogy fails the usual reason is that "
            "the result never left the term it started from."
        ),
    ),
    Term(
        id="candidate-pool",
        label="candidate pool",
        topic="Word sets",
        short=(
            "The words an answer was allowed to come from. GloVe searches all "
            "50,000; the others search only what you have indexed."
        ),
        detail=(
            "This changes what an answer means, so it is printed next to every "
            "one. `queen` from GloVe is a real claim: it is the nearest point in "
            "the whole space. The nearest word from a model with no local "
            "vocabulary is only the nearest of the words you happen to have "
            "added, which is a much weaker claim. Seeding the index is what "
            "makes the second closer to the first."
        ),
    ),
    Term(
        id="out-of-vocabulary",
        label="no vector",
        topic="Word sets",
        short=(
            "A word the model has no entry for. Shown dashed rather than "
            "silently dropped."
        ),
        detail=(
            "GloVe is a fixed 50,000-word English list from 2014, so it has "
            "nothing for `kubernetes`, and nothing for 28 of the 48 words in the "
            "translations grid. A vocabulary has a cutoff date like anything "
            "else. Where too much of a word set is missing, the alignment report "
            "refuses to score it rather than returning a flattering number "
            "computed over the words that survived."
        ),
    ),
]

TERMS_BY_ID: dict[str, Term] = {term.id: term for term in TERMS}


def topics() -> list[str]:
    """Topic names in definition order, for grouping the reference list.

    Order is the order terms were written, which runs from the vectors
    themselves outward to the word sets. Alphabetical would be worse: nobody
    looks up "stress" by its initial letter.
    """
    seen: dict[str, None] = {}
    for term in TERMS:
        if term.topic:
            seen.setdefault(term.topic, None)
    return list(seen)
