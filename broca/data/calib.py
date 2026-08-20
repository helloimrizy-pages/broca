"""Calibration and evaluation token sets, cached to disk on first use.

Every phase reads token IDs from the same cache files, so the calibration set is
byte-identical across the diagnostic, the trace, the sensitivity profile and the
final evaluation.  The cache key includes the dataset, split, tokenizer, sequence
length, count and seed, and each cache carries a SHA-256 of the token array so a
silent change is detectable.

Sampling protocol (C4)
----------------------
Documents are drawn in a seeded random order from one shard of the C4 ``en``
validation split.  A document is used only if it tokenises to at least
``seq_len`` tokens, and a contiguous crop of exactly ``seq_len`` tokens is taken
from a seeded random offset.  This is the GPTQ-style calibration protocol.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..utils import CACHE_DIR

C4_REPO = "allenai/c4"
C4_VALIDATION_SHARD = "en/c4-validation.00000-of-00008.json.gz"
WIKITEXT_REPO = "Salesforce/wikitext"


@dataclass(frozen=True)
class CalibSpec:
    """Everything that determines which tokens come out."""

    dataset: str          # "c4" | "wikitext2"
    split: str            # "validation" | "train" | "test"
    n_sequences: int
    seq_len: int
    seed: int
    tokenizer_name: str

    def cache_stem(self) -> str:
        h = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.dataset}_{self.split}_n{self.n_sequences}_L{self.seq_len}_s{self.seed}_{h}"


def _c4_documents() -> list[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(C4_REPO, C4_VALIDATION_SHARD, repo_type="dataset")
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            docs.append(json.loads(line)["text"])
    return docs


def _wikitext_documents(split: str) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split=split)
    return [t for t in ds["text"]]


def _crop_sequences(docs: list[str], tokenizer, spec: CalibSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    order = rng.permutation(len(docs))
    seqs = []
    for idx in order:
        if len(seqs) >= spec.n_sequences:
            break
        text = docs[int(idx)]
        if not text or len(text) < spec.seq_len:  # cheap length prefilter
            continue
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        if len(ids) < spec.seq_len:
            continue
        start = int(rng.integers(0, len(ids) - spec.seq_len + 1))
        seqs.append(np.asarray(ids[start:start + spec.seq_len], dtype=np.int32))
    if len(seqs) < spec.n_sequences:
        raise RuntimeError(
            f"only found {len(seqs)} documents of at least {spec.seq_len} tokens, "
            f"needed {spec.n_sequences}"
        )
    return np.stack(seqs)


def _contiguous_sequences(docs: list[str], tokenizer, spec: CalibSpec) -> np.ndarray:
    """Join the corpus and cut non-overlapping windows: the standard perplexity protocol."""
    text = "\n\n".join(docs)
    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    n = min(spec.n_sequences, len(ids) // spec.seq_len)
    arr = np.asarray(ids[: n * spec.seq_len], dtype=np.int32).reshape(n, spec.seq_len)
    return arr


def load_tokens(spec: CalibSpec, tokenizer, contiguous: bool = False) -> tuple[np.ndarray, dict]:
    """Return ``(token_ids [n, L], metadata)``, reading the disk cache if present."""
    cache_dir = CACHE_DIR / "tokens"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.cache_stem() + ("_contig" if contiguous else "")
    npy = cache_dir / f"{stem}.npy"
    meta_path = cache_dir / f"{stem}.json"

    if npy.exists() and meta_path.exists():
        tokens = np.load(npy)
        meta = json.loads(meta_path.read_text())
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError(f"token cache {npy} does not match its recorded sha256")
        meta["cache_hit"] = True
        return tokens, meta

    if spec.dataset == "c4":
        docs = _c4_documents()
        source = f"{C4_REPO}:{C4_VALIDATION_SHARD}"
    elif spec.dataset == "wikitext2":
        docs = _wikitext_documents(spec.split)
        source = f"{WIKITEXT_REPO}:wikitext-2-raw-v1:{spec.split}"
    else:
        raise ValueError(f"unknown dataset {spec.dataset!r}")

    tokens = (_contiguous_sequences if contiguous else _crop_sequences)(docs, tokenizer, spec)
    meta = {
        **asdict(spec),
        "source": source,
        "n_documents_available": len(docs),
        "contiguous": contiguous,
        "shape": list(tokens.shape),
        "sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "cache_hit": False,
        "cache_path": str(npy),
    }
    np.save(npy, tokens)
    meta_path.write_text(json.dumps(meta, indent=2))
    return tokens, meta
