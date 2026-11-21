#!/usr/bin/env python3
"""Download the local multilingual encoders the `onnx` sources read.

Four models, ~5.2 GB of weights, into data/models/. They are what the
translations word sets are for: one general multilingual retriever and three
models trained with cross-lingual retrieval as an explicit objective, so the
difference between "has seen many languages" and "was told which strings mean
the same thing" is something you can measure rather than assume.

    python scripts/fetch_models.py                          # all four
    python scripts/fetch_models.py --model minilm-multilingual   # 471 MB
    python scripts/fetch_models.py --list

Nothing is downloaded at request time. app/services/encoders.py opens the
cache with local_files_only=True, so a model that was never fetched makes its
source report itself unavailable — with this command in the message — instead
of pulling 1.9 GB inside a request handler and timing out.

Only the files that are actually used are fetched: the ONNX graph, the fast
tokenizer, for LaBSE the 2 MB Dense layer its sentence-transformers config
declares but its ONNX export leaves out, and for bge-m3 the external-data
sidecar its weights spill into once the graph passes 2 GB. The torch weights
sitting next to them in the same repos are three times the size and this app
has no torch to read them with.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.encoders import (  # noqa: E402
    SPECS,
    SPECS_BY_SOURCE,
    OnnxSpec,
    local_dir,
)
from app.settings import settings  # noqa: E402


def fetch(spec: OnnxSpec) -> int:
    from huggingface_hub import hf_hub_download

    target = local_dir(spec)
    print(f"{spec.source_id}  ({spec.repo}) -> {target.name}/")
    total = 0
    for filename in spec.files:
        # local_dir rather than cache_dir: see encoders.local_dir for why the
        # hub's symlinked blob layout cannot hold a model with external data.
        path = Path(
            hf_hub_download(spec.repo, filename, local_dir=str(target))
        )
        size = path.stat().st_size
        total += size
        print(f"  {size / 1e6:8.1f} MB  {filename}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        choices=[s.source_id for s in SPECS],
        help="only this model; repeatable. Default: all of them.",
    )
    ap.add_argument("--list", action="store_true", help="show what is on disk")
    args = ap.parse_args()

    wanted = (
        [SPECS_BY_SOURCE[m] for m in args.model] if args.model else list(SPECS)
    )

    if args.list:
        from app.services.encoders import spec_available

        print(f"cache: {settings.models_dir}")
        for spec in SPECS:
            state = "present" if spec_available(spec) else "not fetched"
            print(f"  {spec.source_id:24} {spec.dims:4}d  {state:11}  {spec.repo}")
        return 0

    free = shutil.disk_usage(settings.data_dir.parent).free
    # bge-m3 alone is 2.3 GB and the failure mode of running out halfway is a
    # half-written blob that the cache will happily serve as complete.
    if free < 8e9 and not args.model:
        print(
            f"Only {free / 1e9:.1f} GB free and the four models want ~5.2 GB. "
            "Fetch one at a time with --model, or clear some space.",
            file=sys.stderr,
        )
        return 1

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for spec in wanted:
        total += fetch(spec)
    print(f"\n✓ {len(wanted)} model(s), {total / 1e6:.0f} MB in {settings.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
