#!/usr/bin/env python3
"""Precompute the offline demo bundle the frontend falls back to.

Demo mode exists so the deployed site is not a dead page when there is no
backend behind it — which is the normal case for a Vercel deployment of a UI
whose API lives on someone's laptop. It has to be a real demo rather than a
screenshot: every preset, every method and every source switchable, with the
true similarities intact, because those toggles *are* the app.

What it cannot do is embed a word the user typed. That needs a model. So the
bundle covers the curated word sets exhaustively and the UI disables the input
box with a note pointing at the backend.

Writes one JSON per (source, preset) plus an index, into the UI repo:

    python scripts/export_demo.py --out ../visually_sorting_embeddings_ui/public/demo
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.glossary import TERMS  # noqa: E402
from app.data.multilingual import LEXICONS, lexicon_by_id  # noqa: E402
from app.data.presets import ANALOGIES, PRESETS, SAMPLER  # noqa: E402
from app.services import alignment, similarity, store, translate  # noqa: E402
from app.services.embedding import (  # noqa: E402
    NATIVE_SOURCES,
    SOURCES,
    Source,
    embed,
    key_for,
    pivot_of,
    source_status,
)
from app.services.projection import (  # noqa: E402
    METHODS,
    ProjectionError,
    cluster,
    project,
)

# k-means is cheap and the slider has to keep working offline, so ship every
# colouring the UI can ask for rather than freezing one.
CLUSTER_RANGE = range(2, 9)


async def vectors_for(source, words: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    keys: list[str] = []
    for word in words:
        key = key_for(source, word)
        if key and key not in keys:
            keys.append(key)
    have = store.fetch(source, keys)
    todo = [k for k in keys if k not in have]
    missing: list[str] = []
    if todo:
        fresh, missing = await embed(source, todo)
        embedded = [k for k in todo if k not in missing]
        if embedded:
            store.upsert(source, embedded, fresh)
            have.update(dict(zip(embedded, fresh)))
    kept = [k for k in keys if k in have]
    if not kept:
        return [], np.zeros((0, source.dims), dtype=np.float32), missing
    return kept, np.vstack([have[k] for k in kept]), missing


def _labels(
    source: Source, lexicon_id: str, words: list[str]
) -> dict[str, dict[str, str]]:
    """The language and concept of each word, keyed as the points are."""
    if not lexicon_id:
        return {}
    lexicon = lexicon_by_id(lexicon_id)
    if lexicon is None:
        return {}
    on_screen = set(words)
    out: dict[str, dict[str, str]] = {}
    for word, language, concept in lexicon.entries():
        key = key_for(source, word)
        if key in on_screen:
            out[key] = {"language": language, "concept": concept}
    return out


async def _alignment_for(source: Source, lexicon) -> tuple[dict, list[str]]:
    """The alignment report for one source over one grid, for the bundle.

    A near-copy of routes._measure, and deliberately not an import of it: that
    one raises HTTPException, which has no business in a script, and it is six
    lines.
    """
    words, matrix, missing = await vectors_for(source, lexicon.words())
    by_key = {word: matrix[i] for i, word in enumerate(words)}
    entries: list[alignment.Entry] = []
    rows: list[np.ndarray] = []
    for word, language, concept in lexicon.entries():
        key = key_for(source, word)
        if key in by_key and not any(e.text == key for e in entries):
            entries.append(
                alignment.Entry(text=key, language=language, concept=concept)
            )
            rows.append(by_key[key])
    if not entries:
        return {
            "measurable": False,
            "reason": f"{source.label} could not embed any of this word set.",
            "words": 0,
            "concepts": 0,
            "languages": 0,
        }, missing
    return alignment.evaluate(entries, np.vstack(rows)), missing


async def build(out_dir: Path, only_source: str | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"sources": [], "presets": [], "methods": [], "analogies": []}
    written = 0

    usable_sources = []
    for source in SOURCES:
        ok, why = source_status(source)
        if not ok:
            print(f"skip {source.id}: {why}")
            continue
        if only_source and source.id != only_source:
            continue
        usable_sources.append(source)

    for source in usable_sources:
        index["sources"].append(
            {
                "id": source.id,
                "label": source.label,
                "provider": source.provider,
                "model": source.model,
                "dims": source.dims,
                "blurb": source.blurb,
                "supports_phrases": source.supports_phrases,
                "good_at_analogies": source.good_at_analogies,
                "cross_lingual": source.cross_lingual,
                "cross_lingual_note": source.cross_lingual_note,
                "languages": source.languages,
                "pivot": source.pivot,
                "base": source.base,
            }
        )

    index["methods"] = [
        {
            "id": m.id,
            "label": m.label,
            "blurb": m.blurb,
            "detail": m.detail,
            "params": m.params,
            "deterministic": m.deterministic,
            "min_points": m.min_points,
        }
        for m in METHODS
    ]
    phrase_sources = [s.id for s in SOURCES if s.supports_phrases]
    index["presets"] = [
        {
            "id": p.id,
            "label": p.label,
            "blurb": p.blurb,
            "look_for": p.look_for,
            "words": p.words,
            "sources": phrase_sources if p.requires_phrases else [],
            "lexicon": p.lexicon,
        }
        for p in PRESETS
    ]
    index["lexicons"] = [
        {
            "id": lexicon.id,
            "label": lexicon.label,
            "languages": [
                {
                    "code": language.code,
                    "label": language.label,
                    "endonym": language.endonym,
                    "script": language.script,
                }
                for language in lexicon.languages
            ],
            "concepts": [
                {"id": concept.id, "gloss": concept.gloss}
                for concept in lexicon.concepts
            ],
        }
        for lexicon in LEXICONS
    ]
    # The explanations, so the deployed site can explain itself. They are the
    # cheapest thing in the bundle and the easiest to forget: the reference
    # panel renders an empty list without them, which looks like a bug rather
    # than a missing export.
    index["glossary"] = [
        {
            "id": t.id,
            "label": t.label,
            "short": t.short,
            "detail": t.detail,
            "topic": t.topic,
        }
        for t in TERMS
    ]
    index["sampler"] = SAMPLER

    for source in usable_sources:
        for preset in PRESETS:
            if preset.requires_phrases and not source.supports_phrases:
                continue
            # A pivot index only gets a file for the cross-lingual grids.
            # Translating "king" and "queen" to English is a no-op, so a pivot
            # file for the English-only word sets would be a byte-for-byte copy
            # of the native one — 60 duplicated files to say nothing.
            if source.pivot and not preset.lexicon:
                continue
            started = time.time()
            words, matrix, missing = await vectors_for(source, preset.words)
            if len(words) < 3:
                print(f"  {source.id}/{preset.id}: only {len(words)} words, skipping")
                continue

            sims = similarity.cosine_matrix(matrix)
            payload: dict = {
                "source": source.id,
                "preset": preset.id,
                "words": words,
                "labels": _labels(source, preset.lexicon, words),
                "missing": missing,
                "matrix": [[round(float(v), 4) for v in row] for row in sims],
                "top_pairs": similarity.top_pairs(words, sims),
                "clusters": {
                    str(k): cluster(matrix, k)
                    for k in CLUSTER_RANGE
                    if k <= len(words)
                },
                "projections": {},
            }

            # Words that share a vector, worked out once for the whole file.
            # The note depends on the method too, so it is built per method
            # below — but the detection does not.
            identical = similarity.identical_groups(words, matrix)

            for method in METHODS:
                if len(words) < method.min_points:
                    continue
                params = {p["id"]: p.get("default") for p in method.params}
                # The raw-dimension defaults are 0/1/2 for every source, which
                # is the one honest choice: any other triple would imply some
                # dimensions are more meaningful than others.
                try:
                    coords, meta = project(
                        source_id=source.id,
                        method_id=method.id,
                        words=words,
                        vectors=matrix,
                        params=params,
                    )
                except ProjectionError as exc:
                    print(f"  {source.id}/{preset.id}/{method.id}: {exc}")
                    continue
                meta.pop("cached", None)
                drawn = len({
                    (round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5))
                    for c in coords
                })
                payload["projections"][method.id] = {
                    "points": [
                        [round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5)]
                        for c in coords
                    ],
                    # Shipped as the finished sentence rather than as data for
                    # the browser to phrase, so the offline note and the live
                    # one cannot drift apart.
                    "notes": [
                        n
                        for n in [
                            similarity.describe_identical_groups(
                                identical,
                                len(words),
                                method.label,
                                drawn,
                                translated=bool(source.pivot),
                            )
                        ]
                        if n
                    ],
                    "meta": {
                        k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in meta.items()
                    },
                    "params": params,
                }

            target = out_dir / f"{source.id}__{preset.id}.json"
            target.write_text(json.dumps(payload, separators=(",", ":")))
            written += 1
            print(
                f"  wrote {target.name} "
                f"({len(words)} words, {len(payload['projections'])} methods, "
                f"{target.stat().st_size / 1024:.0f} KB, {time.time() - started:.1f}s)"
            )

    # Analogies, precomputed per source. The arithmetic needs a vocabulary to
    # search and the browser has neither the vectors nor the 50k GloVe words,
    # so the answers ship as answers.
    for source in usable_sources:
        for analogy in ANALOGIES:
            await vectors_for(source, [w for w, _ in analogy.terms])
            result = similarity.arithmetic(source, list(analogy.terms), 8)
            index["analogies"].append(
                {
                    "source": source.id,
                    "label": analogy.label,
                    "terms": [{"word": w, "sign": s} for w, s in analogy.terms],
                    "expect": analogy.expect,
                    "note": analogy.note,
                    "result": [
                        {"word": h["word"], "similarity": round(float(h["similarity"]), 4)}
                        for h in result["result"]
                    ],
                    "term_similarities": [
                        {
                            "word": t["word"],
                            "sign": t["sign"],
                            "similarity_to_result": round(
                                float(t["similarity_to_result"]), 4
                            ),
                        }
                        for t in result["terms"]
                    ],
                    "pool": result["pool"],
                    "unknown": result["unknown"],
                }
            )

    # The cross-lingual reports. These are the one part of the bundle that is
    # not a projection: they are properties of the model measured on the full
    # vectors, so they are identical whichever method the user is looking at,
    # and precomputing them is what lets the deployed site answer "does this
    # model sort by meaning or by language" with no backend at all.
    index["alignment"] = {}
    index["pivot"] = []
    for lexicon in LEXICONS:
        for source in usable_sources:
            if lexicon.requires_phrases and not source.supports_phrases:
                continue
            report, missing = await _alignment_for(source, lexicon)
            index["alignment"][f"{source.id}__{lexicon.id}"] = {
                "source": source.id,
                "lexicon": lexicon.id,
                "report": report,
                "missing": missing,
            }
        # And the head-to-head, for every native source whose twin was also
        # exported. The verdict is computed here rather than in the browser so
        # that the deployed site and the live API cannot disagree about it.
        for native in NATIVE_SOURCES:
            twin = pivot_of(native)
            if native not in usable_sources or twin not in usable_sources:
                continue
            if lexicon.requires_phrases and not native.supports_phrases:
                continue
            native_key = f"{native.id}__{lexicon.id}"
            pivot_key = f"{twin.id}__{lexicon.id}"
            if native_key not in index["alignment"] or pivot_key not in index["alignment"]:
                continue
            native_report = index["alignment"][native_key]["report"]
            pivot_report = index["alignment"][pivot_key]["report"]
            index["pivot"].append({
                "lexicon": lexicon.id,
                "native_source": native.id,
                "pivot_source": twin.id,
                "native": native_report,
                "pivot": pivot_report,
                "comparison": alignment.compare(native_report, pivot_report),
            })

    # Every translation the pivot indexes were built from, so the UI can show
    # which words a machine guessed at rather than looked up.
    translations = await translate.to_english(
        [w for lexicon in LEXICONS for w in lexicon.words()]
    )
    index["translations"] = [
        {"text": t.source_text, "english": t.english, "via": t.via}
        for t in translations.values()
    ]

    # Neighbour lists over the full vocabulary, for the inspector panel. Only
    # GloVe can answer these offline in a way worth shipping — for the OpenAI
    # sources the pool is whatever is indexed, which the matrix already covers.
    index["neighbours"] = {}
    glove_source = next((s for s in usable_sources if s.provider == "glove"), None)
    if glove_source is not None:
        vocab_words: list[str] = []
        for preset in PRESETS:
            vocab_words.extend(preset.words)
        vocab_words.extend(SAMPLER)
        seen: set[str] = set()
        for word in vocab_words:
            key = key_for(glove_source, word)
            if key in seen:
                continue
            seen.add(key)
            try:
                hits, _ = similarity.neighbours_of(glove_source, key, 8)
            except KeyError:
                continue
            index["neighbours"][key] = [
                {"word": h["word"], "similarity": round(float(h["similarity"]), 4)}
                for h in hits
            ]

    index["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    total = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    print(f"\n✓ {written} scene files + index.json — {total / 1024:.0f} KB total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("../visually_sorting_embeddings_ui/public/demo"),
    )
    # For iterating on one source's scene files. Note that index.json is
    # always written from the sources this run considered, so a --source run
    # leaves an index that names only that one: the last full run is what the
    # deployed bundle needs, and adding a source to the demo means re-running
    # without this flag.
    ap.add_argument(
        "--source",
        default=None,
        help="only this source id (leaves a partial index.json behind)",
    )
    args = ap.parse_args()
    return asyncio.run(build(args.out.resolve(), args.source))


if __name__ == "__main__":
    raise SystemExit(main())
