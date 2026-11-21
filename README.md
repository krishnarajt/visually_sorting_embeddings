# visually sorting embeddings

The backend for a tool that answers one question honestly: **what does a word
embedding actually look like?**

Embeddings have 300 to 3072 dimensions. Screens have three. Every picture of an
embedding space is therefore a lie of some particular kind, and the interesting
part is *which* kind. This service serves the same words flattened six
different ways, and measures how much each flattening distorted — so the
visualisation can show its own error bars.

The frontend lives in
[visually_sorting_embeddings_ui](https://git.krishnarajthadesar.in/krishnaraj/visually_sorting_embeddings_ui).

## The one design rule

**Positions come from the projection. Every number comes from the full
vectors.**

If the similarities on screen were read off the 3-D coordinates, the plot would
be confirming itself: two words would be "similar" because a projection put
them together. So `app/services/projection.py` only ever places dots, and
`app/services/similarity.py` — neighbours, the pairwise matrix, the arithmetic
— works on the original 300 or 1536 dimensions and never sees a coordinate.

That split is what makes the neighbour edges in the UI worth drawing: an edge
can run clear across the screen, and when it does, you have caught the
projection out.

## Running it

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# GloVe, once. Downloads 820 MB, keeps 55 MB, deletes the rest.
.venv/bin/python scripts/fetch_glove.py

# The three local multilingual encoders, once. 2.8 GB into data/models/.
# Optional, and --model takes one at a time if that is too much at once.
.venv/bin/python scripts/fetch_models.py

# Both optional; GloVe works without either, and each key only affects its
# own sources.
printf 'OPENAI_KEY=sk-...\nCOHERE_KEY=...\n' > .env
PORT=1184 .venv/bin/python main.py
```

Then <http://localhost:1184/docs>. Nothing else is required — no Postgres, no
migrations. Chroma writes to `data/chroma/` on first use.

A half-configured server is a working server, and with eight sources there is
more of it to half-configure: with no OpenAI key the two OpenAI sources report
themselves unavailable *with the reason* and the other six still serve; with no
Cohere key, one source goes; with no GloVe file, the reverse; with no fetched
weights the four local models go and the rest stay. The UI greys out what it
cannot use and says why. The translate-to-English twins need no key of their
own for the built-in word sets, because those carry their own exact
translations.

## The eight sources

Each gets its own Chroma collection, because a 300-d GloVe vector and a 1536-d
OpenAI vector for the same word live in unrelated coordinate systems — cosine
between them is not a weaker similarity, it is a type error. Collections have a
fixed dimensionality, so the database enforces that rather than leaving it to
care.

They are ordered by how each one came by its multilingual ability, because that
turns out to predict its behaviour better than its size does.

| Source | Dims | Cross-lingual | What it is good for |
|---|---|---|---|
| `glove-300` | 300 | none | Stanford, 2014. Single words only. **The analogies work.** |
| `openai-3-small` | 1536 | multilingual | Modern default. Handles whole sentences. |
| `openai-3-large` | 3072 | multilingual | Better at retrieval, twice as much to flatten. |
| `cohere-embed-v3-multilingual` | 1024 | multilingual | 100+ languages, and the one model that asks what the vector is *for*. |
| `e5-multilingual-small` | 384 | multilingual | 100 languages, local, never shown a translation pair. |
| `minilm-multilingual` | 384 | **aligned** | Distilled so a translation lands where the English was. |
| `bge-m3` | 1024 | **aligned** | 2024, 568M parameters, 8192-token context. The modern retriever. |
| `labse` | 768 | **aligned** | 6 billion translation pairs. The reference for this. |

GloVe is here for one reason: `king − man + woman` really does return `queen`
on it (0.69), and really does not on a modern sentence encoder unless you give
it a decent candidate pool. Five of the six built-in analogies land exactly;
`bigger − big + small` answers `larger` by 0.002 ahead of `smaller`, which is
left in because a near-miss you can see is worth more than five hits.

GloVe is also from 2014, so it has no vector for `kubernetes` or `docker`. The
UI shows those chips dashed. A vocabulary has a cutoff date like anything else.

### The input type, which is Cohere's question and nobody else's

`embed-multilingual-v3.0` takes a required `input_type`, and it is not
metadata. `search_query` and `search_document` are trained to be read against
*each other* — a three-word query and a three-paragraph document have to land
together despite looking nothing alike — while `classification` and
`clustering` are for comparing items drawn from one pool. The same string comes
back as a different vector depending on which you asked for.

Every number in this app is the second kind: the similarity matrix, k-means,
the silhouette, the translation-retrieval accuracy. All of them are cosines
between two things from the same set. So the source asks for `clustering` and
asks for nothing else, and that is recorded on `Source.input_type` rather than
read from the environment — the same standing as the pooling in
`encoders.py` and the `query: ` prefix `multilingual-e5` was trained with.
Two input types in one collection would be two subtly different geometries in
one index, and no error would ever be raised about it.

It is worth noticing that the other seven models do not ask because they
cannot, not because they have answered. A single-mode API is a quieter
assumption, not an absent one.

The rest of the Cohere integration is deliberately the same shape as the
OpenAI one — one `httpx` POST per batch, the API's own error text passed
through — with two more differences that belong to the API. 96 texts per call
is a hard maximum, so `COHERE_BATCH_SIZE` is clamped to it and raising
`EMBED_BATCH_SIZE` cannot widen it. And the returned rows carry no index, only
a documented promise of input order, so there is nothing to sort by; the length
of each batch is checked instead, because the failure that assumption would
produce is every vector filed under the wrong word, silently.

### The four local models, and why they cost no new dependency

`e5-multilingual-small`, `minilm-multilingual`, `bge-m3` and `labse` are real
transformers running on your CPU, and getting them here through
sentence-transformers would have meant torch — roughly 2 GB of wheel to obtain
one forward pass. `onnxruntime`, `tokenizers` and `huggingface_hub` were
already installed, because chromadb depends on all three, so
`app/services/encoders.py` runs the ONNX exports directly and the marginal
dependency cost is zero. requirements.txt now names them as direct
dependencies rather than inheriting them, since chromadb is free to drop any of
them.

The cost of that route is that an ONNX export from a sentence-transformers repo
is the *transformer only*: it returns `last_hidden_state` and no pooled vector.
So pooling is ours to get right, and getting it wrong produces numbers that
still look plausible. Each spec transcribes the pooling its `modules.json`
declares, and LaBSE needs one extra piece — its fourth module is a 768×768
Dense layer with a tanh, which lives outside the graph in a 2 MB safetensors
file. Skipping it leaves the model working and measurably blunter: the
same-meaning gap on the translations grid falls from +0.394 to +0.299, because the
margin loss was trained through that layer rather than in front of it —
retrieval stays at 100% and the silhouette by concept does not move, so what
that layer buys is calibration rather than ranking. The
safetensors format is a length-prefixed JSON header and a flat buffer, so it is
parsed with numpy in a dozen lines instead of adding a package.

`bge-m3` needs one more thing. At 568M parameters its graph does not fit in a
single protobuf, so the exporter spills the tensors into a `model.onnx_data`
sidecar and leaves the `.onnx` holding references to it. onnxruntime resolves
those relative to the graph's own directory and then checks the result has not
escaped it — a real protection against a malicious model file — and
huggingface_hub's cache stores blobs content-addressed under `blobs/<2 hex>/`
with symlinks pointing at them, so the graph and its weights land in two
different directories and the check fails:

    External data path escapes model directory.
    resolved: .../blobs/8b/8b3c...   allowed: .../blobs/21

So the weights are fetched with `local_dir` instead and land as ordinary files
in the layout the repo has them in, where the graph and the sidecar are
genuinely siblings. It costs nothing — there is no revision history here to
deduplicate against — and it buys a read path that is one `exists()` call with
no dependency on huggingface_hub at all. Only fetching needs the hub now.

Weights are never downloaded during a request. `scripts/fetch_models.py`
fetches them, exactly as `fetch_glove.py` fetches GloVe, and until it has run
those four sources report themselves unavailable with that command in the
message. They are also not warmed at startup the way UMAP is: UMAP's first
call wastes twelve seconds on a JIT that every later call reuses, so paying it
up front is free, whereas these are 5.2 GB of weights and a visitor who only
looks at GloVe should not have to hold LaBSE in memory to do it.

## Translate first, or trust the model

Every one of those eight has a twin, `<id>-via-en`, which embeds the **English
translation** of the text and files it under the original. That is the other
way to build multilingual search: translate the documents to English,
translate the query to English, and run an ordinary English index. It is how
most of this worked before cross-lingual encoders were good, and the
comparison is not obvious in either direction.

- **native** keeps everything the model knows about the original wording and
  costs one embedding call. It is only as cross-lingual as the model is, which
  for GloVe is not at all.
- **pivot** throws the original wording away and inherits the translator's
  mistakes — but afterwards every vector is an English vector, so *any* English
  model retrieves across languages. `GET /api/neighbours?source=glove-300-via-en&word=собака`
  returns `dogs, cat, pet, puppy, hound` out of the full 50k English
  vocabulary. A 2014 English-only word model, answering a Russian query.

A pivot twin is a separate source with a separate collection, because the
vector filed under `собака` there is the vector of `dog`, and mixing those two
indexes is the same type error the one-collection-per-source rule exists to
prevent. The UI shows eight models and a switch rather than sixteen rows.

Translation comes from the word sets' own English glosses where it can — exact,
free, offline, identical on every run — and from `gpt-4o-mini` otherwise, cached
to `data/translations.json`. Every translation is reported with the translator
that produced it, because a pivot index contains its translator's errors and
which words those are is not guessable. Sentence grids are deliberately
excluded from the glossary: a dictionary translates words, and the grid's
English sentence sits right next to its translations, so a glossary built from
it would hand the pivot index the answer.

### What the comparison actually found

**For single words, the pivot index wins by deleting the question.** A
dictionary turns `eau`, `agua`, `Wasser`, `вода` and `पानी` all into the string
`water`; the model returns one vector for one string; six documents become one.
Retrieval accuracy goes to 100% and concept cohesion to exactly 1.000 because
there is nothing left to retrieve across. That is not skill, it is arithmetic,
and `/api/pivot` says so rather than reading the numbers out — the verdict comes
back `pivot, definitionally`, decided on `collapse`, with the count of concepts
that fused. The finding is real and it cuts both ways: translation does erase
the language, and it erases the difference between the documents with it. That
is perfect recall bought with precision nobody counted.

**For sentences it is a real contest, and much closer than the model cards
suggest.** Translations of a sentence are not byte-identical, so the pivot
index keeps 24 distinct vectors and the comparison has something to compare. On
the `parallel` grid both indexes retrieve every translation correctly for every
model; the pivot column wins on margin alone, and by amounts that range from
decisive to nothing:

| Model | native gap | pivot gap | verdict |
|---|---|---|---|
| `openai-3-small` | +0.322 | +0.782 | pivot, on separation |
| `cohere-embed-v3-multilingual` | +0.424 | +0.655 | pivot, on margin |
| `labse` | +0.491 | +0.599 | pivot, on margin |
| `minilm-multilingual` | +0.855 | +0.912 | pivot, on margin |

Read that last row as the summary. The better a model's own cross-lingual
training, the less a translator adds — `minilm-multilingual` gains 0.06 for the
cost of a translation call on every document *and every query*, a second model
in the path, and an error mode it cannot recover from. The models that benefit
are the ones that needed the help.

## Does an embedding store meaning, or language?

This was the question the whole thing got built out to answer, so it gets its
own word sets, its own measurements and its own answer.

The claim people make about embeddings is that words meaning the same thing sit
near each other. That is only true within one language. A monolingual model has
no reason to put `agua` near `water`: it saw `agua` a few hundred times in
English sentences *about* Spanish, so it learned a vector meaning "a
Spanish-looking word". Language is a much stronger signal in text than meaning
is, and a model learns whatever the data makes cheap.

### Three word grids

A flat word list cannot express "these six strings are one concept and these
eight are one language", so `app/data/multilingual.py` holds concept-by-language
grids instead, and every word in one carries both labels.

| Grid | Shape | What it is for |
|---|---|---|
| `translations` | 8 concepts × 6 languages | The main event. Water, dog, house, book, mother, bread, sun, moon. |
| `scripts` | 5 concepts × 8 writing systems | The hard version — no two alike, and a tokeniser sees nothing but characters. |
| `parallel` | 4 sentences × 6 languages | What the sentence encoders were actually built for. |

A fourth set, `homographs`, has no grid: eight strings that are ordinary words
in both English and German meaning unrelated things — `gift`, `brand`, `rat`,
`hell`, `bad`, `kind`, `gut`, `handy` — each with the English word for both
senses. It is the failure no amount of cross-lingual training fixes, because a
model is handed a string and not a language, and `gift` gets one vector to
cover an English present and a German poison.

English wins seven of the eight on GloVe, on `openai-3-small` and on
`minilm-multilingual`, and eight of eight on LaBSE — so the exceptions are
where the interest is. Both of the first two read `kind` as the German for
child rather than the English for caring, and `minilm-multilingual` reads `gut`
as the German for good, 0.83 against 0.23 for intestine. Knowing more languages
gives a bare string more ways to be ambiguous, and nothing in the input can
settle it.

The French `pain` is in the translations grid for the same reason, and turned
out to be the most discriminating single cell in the whole set. Asked for the
nearest word in another language, `openai-3-small`, `openai-3-large` and
`e5-multilingual-small` all answer `pan` and LaBSE answers `хлеб` — meaning
wins. GloVe answers `mother`. And `minilm-multilingual`, which takes the best
separation scores on every grid here, answers `sol`, the Spanish for sun, with
`pan` only third. A model can be the best in the room at holding translations
together and still be the one that trips over a homograph, which is why
`/api/alignment` returns the list of misses next to the percentage: 94% with
`pain` among the misses is a different model from 94% without it.

### Four numbers, because one would mislead

`app/services/alignment.py` computes all of these on the full vectors, never on
the three coordinates, so flipping to t-SNE does not move them. That is exactly
why they are worth printing next to a t-SNE plot.

- **translation P@1** — of the words in other languages, is the nearest one
  this word's own translation? The question a retrieval index lives on.
- **meaning beats language** — every translation pair against every
  same-language pair, counting how often meaning wins. 0.5 is a coin flip.
  Scale-free, so a model with a compressed cosine range is not punished for it.
- **gap** — mean same-concept cosine minus mean same-language cosine. What a
  similarity threshold in a real system gets set against.
- **silhouette by concept vs by language** — cluster the words by each label
  and ask which one the geometry agrees with. The only number here that can
  come out negative, which is the interesting case: it means the clusters are
  real and you have named them wrongly.

Reporting one of these would be easy and would mislead. Retrieval accuracy
alone flatters `e5-multilingual-small`, which ranks every translation first
while putting all 48 words within 0.06 of each other — perfect ranking, useless
calibration. The gap alone punishes any model with a narrow cosine range for
something no user cares about.

### The answer

`translations`, 8 concepts × 6 languages:

| Source | words | P@1 | meaning>language | gap | sil by concept | sil by language |
|---|---|---|---|---|---|---|
| `glove-300` | 20/48 | 55% | 0.42 | **−0.025** | **−0.043** | **+0.063** |
| `openai-3-small` | 48 | 94% | 0.86 | +0.190 | +0.278 | +0.006 |
| `openai-3-large` | 48 | 100% | 0.90 | +0.131 | +0.263 | +0.094 |
| `cohere-embed-v3-multilingual` | 48 | 100% | 0.96 | +0.174 | +0.480 | +0.052 |
| `e5-multilingual-small` | 48 | 100% | 0.95 | +0.060 | +0.404 | −0.004 |
| `bge-m3` | 48 | 100% | 0.97 | +0.229 | +0.493 | −0.077 |
| `minilm-multilingual` | 48 | 94% | 0.94 | **+0.512** | **+0.761** | −0.168 |
| `labse` | 48 | 100% | **0.99** | +0.394 | +0.743 | −0.129 |

**Yes, but only if the model was trained for it, and the training matters far
more than the size.** Reading the table:

- **GloVe sorts by language, not meaning.** Its gap is *negative* and its
  silhouette by concept is negative while by language it is positive — the
  clusters in that space are languages. And 28 of the 48 words have no vector
  at all. The twelve non-English words it does know — `eau`, `agua`, `chien`,
  `maison`, `casa`, `Haus`, `madre`, `pain`, `pan`, `soleil`, `sol`, `luna` —
  are there because they appear in English text, not because GloVe knows any
  French or Spanish; there is nothing in Cyrillic or Devanagari at all.
  Its worst confusion is worth seeing on its own: the nearest non-English word
  to `water` is the Spanish `pan`, meaning bread, at 0.206 — ahead of `agua` at
  0.170. It has learned that these are both foreign words, and stopped there.
- **The alignment-trained pair wins, and the general multilingual models do
  not.** `minilm-multilingual` and `labse` are the two that were shown
  translation pairs as an explicit objective, and they take the top two
  silhouettes on every grid. `labse` is the most reliable ranker (0.99, and
  1.00 on both other grids); `minilm-multilingual` separates the two axes most
  cleanly (+0.512 gap, +0.761 silhouette) — at 384 dimensions and 118M
  parameters, against LaBSE's 768 and 471M. It is also the one that misreads
  `pain` and `gut`, which is the argument for reading the confusion list rather
  than the leaderboard.
- **Bigger is not more language-agnostic.** `openai-3-large` has twice
  3-small's dimensions and *more* language structure left in it (silhouette by
  language +0.094 against +0.006), with a smaller gap. Scale did not make the
  language go away.
- **Cohere is the strongest retriever that still keeps the language.** It is
  not the best model here on any headline number and it is worth being plain
  about that: its mean same-meaning cosine (0.793) is below `e5` (0.914),
  `labse` (0.887) and `minilm-multilingual` (0.885); its gap (+0.174) is fifth
  of eight; its silhouette by concept (+0.480) is fourth. What it is best at is
  narrower and consistent across three different measures — of the four models
  filed here as merely multilingual it separates meaning from language best, by
  concept silhouette (+0.480 against e5's +0.404 and the OpenAI models' +0.278
  and +0.263), by AUC (0.943) and by best-cutoff F1 (0.917). And it does that
  while its silhouette *by language* stays **positive** (+0.052 here, +0.047 on
  `scripts`), where every translation-trained model goes negative. So it finds
  every translation without having been taught to make the language invisible.
  Whether that is a feature depends entirely on whether you wanted language-
  blind search or language-aware search, which is a question about your
  product and not about the model.
- **A high cosine is not a separable one, and Cohere is the clean case.** Its
  absolute similarities run high — 0.793 same-meaning against 0.618
  same-language — which is the same compression that makes `e5`'s +0.060 gap
  look unusable. The difference is that Cohere's compressed range is still
  separable and e5's much less so: AUC 0.943 against 0.933 on `translations`,
  and 0.990 against 0.877 on `scripts`, where e5 is the second-worst model on
  the grid. Two models with similar-looking means, ranked oppositely once you
  ask whether a threshold could actually be set. This is why the gap is never
  reported on its own.
- **The newest and largest model does not win.** `bge-m3` is from 2024, has
  568M parameters and an 8192-token context, and it retrieves every translation
  correctly with a respectable +0.229 margin — comfortably ahead of both OpenAI
  models and far better calibrated than e5. But on the specific question of
  making language invisible it loses to `minilm-multilingual`, a 118M-parameter
  distillation from 2020, on every measure: margin +0.229 against +0.512,
  silhouette by concept +0.493 against +0.761. It is a retrieval model that
  happens to be multilingual; minilm was built to collapse translations onto
  each other and nothing else. Five times the parameters does not buy the thing
  it was not trained for.
- **A narrow gap is not an unusable one.** `e5-multilingual-small` gets 100%
  P@1 and a gap of +0.060 — its mean same-meaning similarity is 0.914 and its
  mean same-language similarity 0.854. That looks like a model you could not
  set a threshold on, and it is not: its best single cutoff is 0.895 and
  reaches F1 0.861, better than either OpenAI model. Its distributions are
  narrow *and* separable, so the gap — a difference of two means — understates
  it. This is the one place the four headline numbers can mislead together,
  and the fix is the section below.
  It is also the only one of the three local models never shown a translation
  pair, and it sits exactly where that predicts: between the OpenAI models and
  the aligned ones.
- **Script is not the obstacle it looks like.** On `scripts` — eight alphabets,
  nothing shared at the character level — `minilm-multilingual` scores +0.919
  by concept and −0.242 by language, *better* than on the mostly-European
  translations grid. Whereas GloVe has 5 of 40 words and the report refuses to
  score it, which is the honest output rather than a flattering number computed
  over the English row.
- **Sentences are the easiest case, not the hardest.** Every model retrieves
  every translation on `parallel`. A sentence carries grammar, tense and topic,
  all of which survive translation, where a bare noun gives the model almost
  nothing to work with.

## Is there a "water dimension"?

No, and the interesting part is how close to yes it gets.

The `raw` projection method has always existed to show that picking three
dimensions to draw is hopeless. It used to only pick them by index, which is a
soft version of the test — someone is entitled to think a better choice would
have worked. It now also picks the three that differ most across exactly the
words on screen, which is the strongest case any axis-aligned triple can make,
and reports what they explain against what three PCA components explain on the
same words:

| | explains |
|---|---|
| dimensions 0, 1, 2 | 0.35% |
| the three most varied of 768 | 1.11% |
| three PCA components | **46.08%** |

Forty-one times better, from three learned combinations instead of three
columns. That ratio is why the other five methods exist, and it is now a number
in the readout rather than an assertion in a paragraph.

But "the dimensions mean nothing" is too strong, and `POST /api/dimensions` is
the other half. The **set** of dimensions a word drives hardest is partly
shared by its translations:

    water / вода    cosine 0.98   shares 5 of its top 5
    water / eau     cosine 0.97   shares 4
    book  / книга   cosine 0.95   shares 3
    book  / Buch    cosine 0.85   shares 2
    book  / livre   cosine 0.65   shares 1

Across the whole grid the rank correlation between overlap and cosine is
positive for every model, and how positive follows the same ordering everything
else here does:

| Source | median | with hub dimensions deleted | hub dims found |
|---|---|---|---|
| `minilm-multilingual` | **+0.69** | +0.54 | 3 |
| `labse` | +0.53 | +0.53 | 1 |
| `glove-300` | +0.51 | +0.37 | 1 |
| `cohere-embed-v3-multilingual` | +0.42 | +0.39 | 1 |
| `e5-multilingual-small` | +0.38 | +0.38 | 0 |
| `openai-3-large` | +0.33 | +0.30 | 2 |
| `bge-m3` | +0.29 | +0.28 | **5** |
| `openai-3-small` | +0.21 | +0.27 | 1 |

The two translation-trained models come top, which is the ordering the
alignment silhouettes give by a completely different route. That agreement is
worth something.

What it is **not** is a similarity measure. A median of +0.5 with per-word
figures scattering from −0.14 to +0.83 means individual words mislead badly,
and cosine already answers "how similar" using all 768 dimensions rather than
five. What the dimension profile adds is *which* dimensions, and shared with
whom, which cosine collapses away.

### Hub dimensions, and the fix that broke it

Ranking by raw magnitude has an obvious objection: dimension 216 is in the top
five of `water`, `dog` *and* `book` on LaBSE, because it is simply large for
almost every vector. These models carry a few such hub dimensions holding a
global offset rather than anything about the word.

The textbook answer is to centre — rank by distance from the mean in units of
that dimension's spread — and it was the first thing implemented here. It
removes the hubs and the signal with them: when the comparison set is a
translation grid, the dimensions a concept's members share sit near the set
mean *by construction*, so what survives centring is mostly each word's own
idiosyncrasy. The median falls from +0.53 to +0.40 on LaBSE and to +0.03 on
GloVe. Centring is the right answer to the hub problem and the wrong answer to
this question.

So hubs are flagged rather than subtracted, and the overlap is also reported
with them deleted, per pair — the right-hand columns above. There is no single
verdict to fold that into: on LaBSE the agreement is untouched, on GloVe it
loses a third, and on `openai-3-small` deleting the hubs *improves* it, because
there they were noise. The hub count is worth seeing on its own too. `bge-m3`
has five where `e5-multilingual-small` has none, a difference between the two
models that nothing else in this app surfaces.

## Every number is explained in the app

`/api/meta` carries a `glossary`: 24 terms across five topics, from `cosine
similarity` and `hub dimension` to `collapse` and `candidate pool`. Each has a
one-sentence form for the label it appears on and a paragraph for the reference
list, and the paragraph is allowed to say how the number misleads — which is
usually the more useful half.

It is served rather than hardcoded in the frontend for the reason `/api/meta`
exists at all: the definition of trustworthiness belongs beside the code that
computes trustworthiness, and a second copy in a React component is a second
thing to keep true. `app/data/glossary.py` is the file; where a term names
something a module computes, that module's docstring is the long-form source
and the glossary is the summary, and if they ever disagree the module is right.

## The six methods

| Method | Keeps | Gives up |
|---|---|---|
| `pca` | global shape, deterministic | most of the variance — it reports how much |
| `tsne` | local neighbourhoods | all distance between clusters |
| `umap` | neighbourhoods and some global shape | arguability |
| `mds` | true cosine distances, measurably | reports its residual as stress |
| `random` | nothing in particular | the control condition |
| `raw` | three literal dimensions | shows single dimensions mean nothing |

Every projection carries two numbers computed against the full-dimensional
vectors:

- **trustworthiness** — of each point's *k* nearest neighbours on screen, how
  many were genuinely among its *k* nearest in full dimensions.
- **distance_correlation** — rank correlation between every true cosine
  distance and the distance actually drawn.

Read together they identify the method. On the six-field word set: PCA scores
0.87 / 0.65, UMAP 0.89 / 0.67, MDS 0.88 / **0.76** (it optimises exactly that),
t-SNE 0.72 / **0.37** — right neighbours, wrong distances, which is t-SNE's
signature and the most misread property of these plots — and random projection
0.53 / **0.01**, which is what no method at all looks like.

## API

`GET /api/meta` is one call that tells the UI everything: which sources have
keys, which methods are importable, the word sets, the analogies. The frontend
keeps no second copy of that.

`POST /api/scene` does everything else in one round trip — embed what is new,
store it, project all of it, measure the distortion, compute the true
similarities. One call per change to the word list is what keeps the points and
the numbers beside them provably from the same request.

`POST /api/alignment` measures one model against one word grid: the four
numbers above, plus a per-language breakdown against English, the concepts that
held together worst, and the list of words whose nearest cross-language
neighbour was the wrong one. That last list is the part worth reading — the
mistakes are almost never random.

`POST /api/dimensions` profiles one word against the others on screen: the
dimensions it drives hardest, which of them are hubs, who shares them, and the
rank correlation between that overlap and cosine. Scores are relative to the
comparison set the caller sends, which the response says, because "unusual
among these words" is a different claim from "unusual in the model".

`POST /api/pivot` runs a model and its translate-to-English twin over the same
grid and says which won, on which measure, and by how much. It refuses to
report a verdict where the pivot index collapsed most of the grid to one string
per concept, and explains that instead.

Then `POST /api/arithmetic`, `GET /api/neighbours`, `POST /api/index/seed`,
`GET /api/index`, `DELETE /api/index`. Details at `/docs`.

### The candidate pool, and why it is printed

Neighbour and analogy answers say which pool they searched, because the two
providers differ in a way that changes what the answer means:

- **GloVe** searches all 50,000 words on disk. `queen` comes back because it is
  genuinely the nearest point in the space.
- **OpenAI** searches only what is indexed — there is no local vocabulary and
  enumerating one would mean embedding 50k words at the API. So the answer is
  the nearest word *you have added*, which is a weaker claim.

`POST /api/index/seed` embeds all 254 built-in words to make that pool worth
searching. It costs a fraction of a cent, and it is the difference between the
OpenAI analogy feature working and not: seeded, `king − man + woman` returns
`queen` at 0.62 there too.

## Deploying

One pod on the same cluster as the other backends, with no database, no volume
and no login. Pushing to `main` runs `.gitea/workflows/build-and-push.yml`,
which builds the image, pushes it to the Gitea registry and commits the new tag
into `k8s/overlays/gitea/kustomization.yaml`; Argo CD syncs that.

Once, by hand:

1. A `REGISTRY_TOKEN` secret on the Gitea repo, as on the other backends.
2. A Vault entry at `apps/vsembeddings/vsembeddings/env` holding `OPENAI_KEY`
   and `COHERE_KEY`. Both are optional — a source whose key is missing is
   greyed out in the UI with the reason — and the pod starts without the
   entry at all.
3. `kubectl apply -f k8s/base/argocd-app.yaml`.
4. A public hostname on the Cloudflare tunnel, `vsembeddings-api.krishnarajthadesar.in`
   → the NodePort, **32051**. That is the address the UI is built to use.

The API is open to the internet, so two things differ from running it locally,
both set in `k8s/base/deployment.yaml`: `CORS_ORIGINS=*`, because there is
nothing a credentialed request could reach that an anonymous one cannot, and
`RATE_LIMIT_PER_MINUTE=120` per client ([`app/ratelimit.py`](app/ratelimit.py)),
because every unseen word typed into a hosted source is a billable embedding
and there is no account to attribute it to. `/health` and `/ready` are never
limited.

### What the image carries, and what it costs

Weights are never downloaded at request time, so they are in the image — GloVe
and all four local encoders, fetched at build time, each in its own stage and
its own registry layer. Totals measured on the built image running in a
container; the per-row image sizes are approximate:

| | Image | Resident memory |
|---|---|---|
| Idle: GloVe, UMAP, Chroma | ≈1.0 GB | 0.45 GB |
| + multilingual-e5-small | +0.5 GB | +0.8 GB |
| + paraphrase-multilingual-MiniLM | +0.5 GB | +0.7 GB |
| + LaBSE | +1.9 GB | +1.9 GB |
| + BGE-M3 | +2.2 GB | +1.4 GB |
| **All four, once each has been used** | **6.1 GB** | **5.1 GB** |

An encoder is loaded by the first request that needs it and then kept, so the
idle pod is the first row and only a visitor who tries every model reaches the
last. The container's memory limit is 7 GiB for that case.

To run lighter, delete a model's stage and its `COPY --from` line from the
[`Dockerfile`](Dockerfile). That source then reports itself unavailable and the
UI greys it out; nothing else changes. The Chroma store under `/app/data/chroma`
is only a cache of embeddings already paid for, so it is not on a volume and a
restart simply empties it.

## Scripts

| Script | What it does |
|---|---|
| `scripts/fetch_glove.py` | Builds `data/glove.6B.300d.npz` from Stanford's zip |
| `scripts/fetch_models.py` | Downloads the three ONNX encoders into `data/models/` |
| `scripts/export_demo.py` | Precomputes the UI's offline bundle (109 files, 2.0 MB) |

`export_demo.py` is what makes the deployed site work with no backend behind
it — the normal state of a Vercel deployment whose API is on a laptop. It
exhaustively covers every (source × preset × method) combination, so offline
the toggles are all live; only typing a new word needs a model. The alignment
reports and the translate-first comparisons ship as answers too, since they are
properties of the model rather than of the picture and do not change when the
user flips a method.

One exception keeps the bundle honest rather than merely smaller: a pivot
source only gets files for the three cross-lingual grids. Translating `king`
and `queen` to English is a no-op, so a pivot file for the English-only word
sets would be a byte-for-byte copy of the native one — sixty duplicated files
to say nothing.

## Notes

- **UMAP is warmed at startup.** Its first call in a process spends ~12 s in
  numba's JIT and every call after that 30 ms. Since UMAP is the default
  method, leaving that cost in place would make the first visit look like a
  broken server.
- **t-SNE and UMAP are seeded**, so the same words give the same picture twice.
  `/api/scene` also accepts an `anchor` of the previous frame's coordinates and
  fits the new layout onto it by orthogonal Procrustes — rotation and
  reflection carry no information about the data, so removing them removes the
  part of a change that is arbitrary and leaves the part that is real.
- **Case folding is a property of the source, not of the app.** GloVe's
  vocabulary is lowercase, so its keys have to be folded; the transformer
  models are cased, and folding `Wasser` to `wasser` hands them a spelling
  German does not use. That is not a rounding error — it costs
  `minilm-multilingual` two of its twenty-four translation retrievals on the
  translations grid, dropping it from 100% to 92%. So ids come from
  `key_for(source, text)` rather than a global normaliser, which is safe
  precisely because no two sources share a collection.
- **Chroma's telemetry logger is silenced** in `main.py`: 0.6.3 ships a
  telemetry call whose signature does not match its own `capture()`, so it logs
  a failure per operation even with telemetry off. Nothing is being sent.
