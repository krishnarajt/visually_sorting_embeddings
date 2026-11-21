"""The local transformer encoders, run on ONNX Runtime.

These are the models that answer the cross-lingual question, so they are here
for a reason rather than for variety: each one came by its multilingual
ability a different way, and the differences show up in the numbers
app/services/alignment.py computes.

  multilingual-e5-small  general multilingual retrieval. Trained contrastively
                         on mined pairs across 100 languages, but never told
                         that two strings are translations of each other.
  minilm-multilingual    multilingual knowledge distillation (Reimers &
                         Gurevych, 2020): a student was trained to put a
                         translated sentence at the *same coordinates* as the
                         English original that a monolingual teacher produced.
                         Cross-lingual alignment is the entire training
                         objective.
  labse                  trained on 6 billion translation pairs with an
                         additive-margin softmax that pushes a sentence and
                         its translation together and everything else apart.
                         The reference model for translation retrieval.
  bge-m3                 the current strong multilingual retriever: 568M
                         parameters of XLM-RoBERTa-large, 100+ languages, and
                         trained on mined *and* parallel data with the
                         explicit goal of cross-lingual retrieval. The most
                         modern answer to the question this app is asking, and
                         the largest model here by a factor of five.

## Why ONNX rather than sentence-transformers

Running these through sentence-transformers would mean torch, and torch is
~2 GB to get a forward pass this file does in fifty lines. onnxruntime,
tokenizers and huggingface_hub are already installed — chromadb depends on all
three — so the marginal dependency cost of three real multilingual encoders
here is zero. requirements.txt now names them directly rather than leaning on
that being true by accident.

The cost of the ONNX route is that the exported graph is the *transformer
only*: it returns `last_hidden_state` and nothing else. Pooling is therefore
ours to get right, and getting it wrong is the kind of mistake that still
produces plausible-looking numbers. Each spec below carries the pooling its
sentence-transformers `modules.json` declares, and `_read_safetensors` exists
solely because LaBSE's fourth module is a 768x768 Dense layer that lives
outside the graph — see `OnnxSpec.dense_file`.

## Weights are not downloaded on demand

`scripts/fetch_models.py` fetches them, exactly as `scripts/fetch_glove.py`
fetches GloVe. A request that would otherwise pull 2.3 GB mid-flight reports
the source as unavailable instead, with the command to run — and since this
module reads plain paths rather than a hub cache, it never has the option.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnnxSpec:
    """Everything needed to reproduce one model's sentence embedding.

    The fields after `repo` are not configuration — they are transcribed from
    that repo's `modules.json` and `1_Pooling/config.json`. Changing one
    silently changes what the model means.
    """

    source_id: str
    repo: str
    dims: int
    # "mean" — attention-mask-weighted mean over tokens, or "cls" — the first
    # token. Whichever `1_Pooling/config.json` sets to true.
    pooling: str
    # A sentence-transformers Dense module applied after pooling, if the model
    # has one. LaBSE does: Linear(768, 768) then tanh. Skipping it leaves the
    # model working and measurably worse calibrated: on the translations grid
    # the same-meaning / same-language gap falls from +0.394 to +0.299, while
    # retrieval stays at 100% and the silhouette by concept does not move
    # (+0.743 against +0.751). So the layer the margin loss was trained through
    # buys separation rather than ranking — which is the part a similarity
    # threshold in a real system depends on.
    dense_file: str = ""
    # e5 was trained with every input prefixed. Without it the model is being
    # used off-distribution; with it the numbers match the published ones.
    query_prefix: str = ""
    onnx_file: str = "onnx/model.onnx"
    tokenizer_file: str = "tokenizer.json"
    # Weights that live beside the graph rather than inside it. Past 2 GB a
    # model cannot fit in one protobuf, so the exporter spills the tensors into
    # a sidecar and leaves the .onnx holding only references to it. Nothing
    # opens these directly — onnxruntime resolves them by name, relative to the
    # graph's own directory — but they have to be on disk before the graph
    # loads, and the availability check has to know they exist or a fetch that
    # got the 200 KB header and none of the 2.3 GB of weights would look
    # complete.
    data_files: tuple[str, ...] = ()
    # Long enough for any sentence in the word sets, short enough that a batch
    # of single words does not pay for padding it will never use.
    max_length: int = 256

    @property
    def files(self) -> list[str]:
        """Every file `scripts/fetch_models.py` has to place on disk."""
        wanted = [self.onnx_file, self.tokenizer_file]
        if self.dense_file:
            wanted.append(self.dense_file)
        wanted.extend(self.data_files)
        return wanted


SPECS: list[OnnxSpec] = [
    OnnxSpec(
        source_id="e5-multilingual-small",
        repo="intfloat/multilingual-e5-small",
        dims=384,
        pooling="mean",
        query_prefix="query: ",
    ),
    OnnxSpec(
        source_id="minilm-multilingual",
        repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dims=384,
        pooling="mean",
    ),
    OnnxSpec(
        source_id="labse",
        repo="sentence-transformers/LaBSE",
        dims=768,
        pooling="cls",
        dense_file="2_Dense/model.safetensors",
    ),
    OnnxSpec(
        source_id="bge-m3",
        repo="BAAI/bge-m3",
        dims=1024,
        # `1_Pooling/config.json` sets cls, and module 2 is Normalize, which
        # embed() already applies to everything. No Dense module and no
        # instruction prefix: bge-m3 dropped the "Represent this sentence
        # for..." convention its English-only siblings still want, and adding
        # one back would put the model off-distribution.
        pooling="cls",
        data_files=("onnx/model.onnx_data",),
    ),
]

SPECS_BY_SOURCE: dict[str, OnnxSpec] = {s.source_id: s for s in SPECS}


# ---------------------------------------------------------------------------
# Weights on disk
# ---------------------------------------------------------------------------


def local_dir(spec: OnnxSpec) -> Path:
    """Where this model's files live: one plain directory per repo.

    Deliberately not huggingface_hub's own cache, which stores blobs
    content-addressed under `blobs/<2 hex>/<sha>` and fills `snapshots/` with
    symlinks to them. That layout is good for deduplicating revisions and
    fatal for a model whose weights do not fit in one protobuf: onnxruntime
    resolves an external-data reference relative to the graph's directory and
    then checks that the result has not escaped it, and two symlinks pointing
    into two different blob subdirectories fail that check —

        External data path escapes model directory.
        resolved: .../blobs/8b/8b3c...  allowed: .../blobs/21

    which is a real protection against a malicious model file, not a bug to be
    worked around. So `scripts/fetch_models.py` asks the hub for `local_dir`
    instead and gets ordinary files in the layout the repo has them in, where
    `onnx/model.onnx` and `onnx/model.onnx_data` are genuinely siblings.

    It costs nothing here — there is no revision history to deduplicate
    against — and it buys a read path that is one `exists()` call with no
    dependency on huggingface_hub at all. Only fetching needs the hub now.
    """
    return settings.models_dir / spec.repo.replace("/", "__")


def _cached_path(spec: OnnxSpec, filename: str) -> Path | None:
    """Where a file sits on disk, or None if it was never fetched."""
    path = local_dir(spec) / filename
    return path if path.exists() else None


def spec_available(spec: OnnxSpec) -> bool:
    """Whether every file this model needs is present.

    Every file, including the external-data sidecar: a fetch interrupted after
    bge-m3's 0.7 MB graph and before its 2.3 GB of weights would otherwise
    look complete and fail at load time with something far less obvious.
    """
    return all(_cached_path(spec, f) is not None for f in spec.files)


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """The two arrays out of a safetensors file, without the safetensors package.

    The format is a little-endian u64 header length, that many bytes of JSON
    describing each tensor's dtype, shape and byte range, then the raw buffer.
    Parsing it here is a dozen lines and keeps a dependency out of
    requirements.txt for the sake of one 2 MB file of float32.
    """
    with path.open("rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
        blob = fh.read()
    out: dict[str, np.ndarray] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != "F32":
            raise ValueError(f"{path.name}: {name} is {entry['dtype']}, expected F32")
        start, end = entry["data_offsets"]
        out[name] = np.frombuffer(blob[start:end], dtype=np.float32).reshape(
            entry["shape"]
        )
    return out


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------


class OnnxEncoder:
    """One model, loaded once per process and held.

    Loading is deferred until the first embed call and guarded by a lock: the
    three models together are 2.8 GB of weights, and a fresh visit that only
    ever touches GloVe should not pay to map them in.
    """

    def __init__(self, spec: OnnxSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()
        self._dense: dict[str, np.ndarray] | None = None

    @property
    def available(self) -> bool:
        return spec_available(self.spec)

    def load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            paths = {f: _cached_path(self.spec, f) for f in self.spec.files}
            missing = [f for f, p in paths.items() if p is None]
            if missing:
                raise FileNotFoundError(
                    f"{self.spec.repo} is not on disk ({', '.join(missing)}). "
                    "Fetch it once with `python scripts/fetch_models.py`."
                )

            logger.info(f"Loading {self.spec.repo} ({self.spec.dims}d)")
            tokenizer = Tokenizer.from_file(str(paths[self.spec.tokenizer_file]))
            tokenizer.enable_truncation(max_length=self.spec.max_length)
            # The pad token differs between the two tokenizer families and the
            # library's default of id 0 is wrong for one of them: 0 is [PAD]
            # for BERT but <s> for XLM-RoBERTa. Padded positions are masked out
            # of both attention and pooling, so the wrong choice would not
            # change any number here — it would just mean the file quietly
            # disagreed with the model about what it was doing.
            pad_id, pad_token = None, None
            for candidate in ("[PAD]", "<pad>"):
                found = tokenizer.token_to_id(candidate)
                if found is not None:
                    pad_id, pad_token = found, candidate
                    break
            if pad_id is None:
                raise ValueError(f"{self.spec.repo}: no pad token in the tokenizer")
            tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

            session = ort.InferenceSession(
                str(paths[self.spec.onnx_file]),
                providers=["CPUExecutionProvider"],
            )
            outputs = [o.name for o in session.get_outputs()]
            hidden = session.get_outputs()[0].shape[-1]
            if isinstance(hidden, int) and hidden != self.spec.dims:
                raise ValueError(
                    f"{self.spec.repo} returns {hidden} dimensions, "
                    f"expected {self.spec.dims}"
                )
            logger.debug(f"  graph outputs: {outputs}")

            if self.spec.dense_file:
                weights = _read_safetensors(paths[self.spec.dense_file])
                self._dense = {
                    "weight": weights["linear.weight"],
                    "bias": weights["linear.bias"],
                }

            self._input_names = {i.name for i in session.get_inputs()}
            self._tokenizer = tokenizer
            self._session = session
            logger.info(f"✓ {self.spec.source_id} ready")

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised sentence embeddings, one row per input, in order."""
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        self.load()
        assert self._tokenizer is not None and self._session is not None

        out: list[np.ndarray] = []
        batch_size = settings.local_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer.encode_batch(
                [self.spec.query_prefix + t for t in batch]
            )
            ids = np.asarray([e.ids for e in encoded], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            # BERT wants them, XLM-RoBERTa's graph has no such input. All of
            # our inputs are single segments either way.
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feed)[0]
            out.append(self._pool(hidden, mask))

        vectors = np.vstack(out)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def _pool(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.spec.pooling == "mean":
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(
                weights.sum(axis=1), 1e-9
            )
        else:
            raise ValueError(f"unknown pooling {self.spec.pooling!r}")
        if self._dense is not None:
            pooled = np.tanh(pooled @ self._dense["weight"].T + self._dense["bias"])
        return pooled.astype(np.float32)


ENCODERS: dict[str, OnnxEncoder] = {s.source_id: OnnxEncoder(s) for s in SPECS}


def encoder_for(source_id: str) -> OnnxEncoder:
    encoder = ENCODERS.get(source_id)
    if encoder is None:
        raise KeyError(f"no local encoder for {source_id!r}")
    return encoder
