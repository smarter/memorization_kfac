"""
Calibration-corpus loading for K-FAC / OBD factor collection.

The default `load_dataset(..., data_files="data/**/*.json*", streaming=True)`
flow consumes shards in alphabetical order, so a small per-byte budget like
the default 100MB is exhausted from the alphabetically-first subdomain
directory only — never seeing OpenWebMath, ArXiv, AlgebraicStack, etc. for
Dolmino. That biases K-FAC's eigenspectrum toward whichever subdomain comes
first lexicographically.

This module exposes ``collect_sequences`` with a ``mix_strategy`` flag that
controls how shards are ordered:

  - ``first_n``     (default): the original alphabetical order. Reproduces
    the legacy bias toward whichever subdomain comes first lexicographically.
  - ``shuffle``    : shuffle the global shard list once with --seed.
  - ``interleave`` : group shards by top-level subdomain, then round-robin
    between subdomains. Within ``--nbytes`` you get a roughly uniform mix
    across subdomains regardless of relative shard sizes.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

from datasets import (
    Features,
    Value,
    interleave_datasets,
    load_dataset,
)
from huggingface_hub import HfApi
from tqdm.auto import tqdm


HF_DATASETS = {
    "olmo": "allenai/olmo-mix-1124",
    "dolmo": "allenai/dolmino-mix-1124",
}


def _list_data_files(repo_id: str) -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return [
        f for f in files
        if f.startswith("data/") and (".json" in f) and not f.endswith("/")
    ]


def _features_for(corpus: str) -> Optional[Features]:
    # The olmo-mix-1124 shards' JSON has fields beyond `text`; pinning the
    # schema avoids HF datasets inferring a richer struct that fails parsing
    # on shards with missing keys.
    if corpus == "olmo":
        return Features({"text": Value("string")})
    return None


def _streaming_ds(repo_id: str, files: Iterable[str], features: Optional[Features]):
    urls = [f"hf://datasets/{repo_id}/{f}" for f in files]
    return load_dataset(
        "json",
        data_files={"train": urls},
        split="train",
        streaming=True,
        features=features,
    )


def _build_iter_dataset(
    corpus: str,
    files: list[str],
    mix_strategy: str,
    seed: int,
):
    repo_id = HF_DATASETS[corpus]
    features = _features_for(corpus)

    if mix_strategy == "first_n":
        return _streaming_ds(repo_id, sorted(files), features)

    if mix_strategy == "shuffle":
        ordered = sorted(files)
        random.Random(seed).shuffle(ordered)
        return _streaming_ds(repo_id, ordered, features)

    if mix_strategy == "interleave":
        # Group by top-level subdomain, e.g. ``data/dclm`` vs ``data/openwebmath``.
        # Round-robin over subdomains rather than over thousands of shards so
        # the per-subdomain stream is itself a sensible "all of subdomain X"
        # iterator and we don't pay metadata cost for thousands of streams.
        by_subdomain: dict[str, list[str]] = {}
        for f in files:
            parts = f.split("/", 2)
            subdomain = parts[1] if len(parts) >= 3 else "_root"
            by_subdomain.setdefault(subdomain, []).append(f)

        rng = random.Random(seed)
        per_subdomain = []
        for subdomain in sorted(by_subdomain):
            sd_files = sorted(by_subdomain[subdomain])
            rng.shuffle(sd_files)
            per_subdomain.append(_streaming_ds(repo_id, sd_files, features))

        if len(per_subdomain) == 1:
            return per_subdomain[0]

        return interleave_datasets(
            per_subdomain,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    raise ValueError(
        f"Unknown mix_strategy: {mix_strategy!r}. "
        "Expected one of {'first_n', 'shuffle', 'interleave'}."
    )


def collect_sequences(
    corpus: str,
    tokenizer,
    seq_len: int,
    nbytes: int,
    *,
    seed: int = 42,
    mix_strategy: str = "first_n",
) -> list[list[int]]:
    """Stream up to ``nbytes`` of raw text from ``corpus`` and tokenise into
    fixed-length ``seq_len`` sequences. See module docstring for
    ``mix_strategy`` semantics."""
    if corpus not in HF_DATASETS:
        raise ValueError(
            f"Unknown corpus {corpus!r}; known: {sorted(HF_DATASETS)}"
        )

    files = _list_data_files(HF_DATASETS[corpus])
    if not files:
        raise SystemExit(f"No data shards found in {HF_DATASETS[corpus]}")

    print(
        f"[calibration] corpus={corpus} mix={mix_strategy} "
        f"shards={len(files)} target_bytes={nbytes:,}"
    )

    ds = _build_iter_dataset(corpus, files, mix_strategy, seed)

    sequences: list[list[int]] = []
    buf: list[int] = []
    seen = 0
    for sample in tqdm(ds, desc="Loading sequences"):
        txt = sample["text"].strip()
        if not txt:
            continue
        seen += len(txt.encode())
        buf.extend(tokenizer(txt, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            sequences.append(buf[:seq_len])
            buf = buf[seq_len:]
        if seen >= nbytes:
            break

    return sequences
