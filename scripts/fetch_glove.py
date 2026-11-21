#!/usr/bin/env python3
"""Build the local GloVe matrix that the glove-300 source reads.

Downloads Stanford's glove.6B.zip (~820 MB), keeps the most frequent N words
from the 300-d file, and writes them to data/glove.6B.300d.npz (~60 MB at the
default 50k words). The zip and the extracted 1 GB text file are deleted
afterwards unless --keep-download is passed.

The trim is free of judgement calls: the GloVe files are written in descending
frequency order, so the first N lines *are* the N most frequent words. What it
buys is a file that loads in under a second instead of one that needs a
gigabyte of RAM to answer "what is near queen".

    python scripts/fetch_glove.py                 # 50k words
    python scripts/fetch_glove.py --vocab 100000  # more, slower, bigger
    python scripts/fetch_glove.py --zip ~/glove.6B.zip   # skip the download
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.settings import settings  # noqa: E402

URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
MEMBER = "glove.6B.300d.txt"


def download(target: Path) -> None:
    import httpx

    print(f"Downloading {URL}")
    print("  ~820 MB. This is the slow part and it only happens once.")
    with httpx.stream("GET", URL, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        done = 0
        with target.open("wb") as fh:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="")
        print()


def parse(text_path: Path, vocab: int) -> tuple[list[str], np.ndarray]:
    words: list[str] = []
    rows: list[np.ndarray] = []
    skipped = 0
    with text_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(words) >= vocab:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            # Punctuation and bare digits have vectors but nothing to say in a
            # visualisation of word meaning, and they occupy the frequent end
            # of the file where the useful words are.
            if not any(ch.isalpha() for ch in word):
                skipped += 1
                continue
            words.append(word)
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    print(f"  kept {len(words)} words, skipped {skipped} non-alphabetic tokens")
    return words, np.vstack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=50_000, help="how many words to keep")
    ap.add_argument("--zip", type=Path, default=None, help="an already-downloaded zip")
    ap.add_argument("--keep-download", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or settings.glove_path
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "_glove_work"
    work.mkdir(exist_ok=True)

    zip_path = args.zip
    downloaded_here = False
    if zip_path is None:
        zip_path = work / "glove.6B.zip"
        if not zip_path.exists():
            download(zip_path)
            downloaded_here = True
    if not zip_path.exists():
        print(f"No such zip: {zip_path}", file=sys.stderr)
        return 1

    print(f"Extracting {MEMBER}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(MEMBER) as src, (work / MEMBER).open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)

    print(f"Parsing the {args.vocab} most frequent words")
    words, vectors = parse(work / MEMBER, args.vocab)

    np.savez_compressed(out, words=np.asarray(words), vectors=vectors)
    print(f"✓ Wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

    (work / MEMBER).unlink(missing_ok=True)
    if downloaded_here and not args.keep_download:
        zip_path.unlink(missing_ok=True)
    if not any(work.iterdir()):
        work.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
