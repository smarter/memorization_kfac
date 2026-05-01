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
  - ``dolmino_50B``: weighted interleave matching the 50B Dolmino mix
    proportions published by AI2 (47.2% DCLM / 16.6% FLAN / 5.85% pes2o /
    7.11% Wiki / 2.45% StackExchange / 20.8% Stage 2 Math). Only valid for
    ``corpus=dolmo``.
  - ``gsm8k``      : restrict the shard list to ``data/math/gsm8k/`` only.
    Use this to ask "does a GSM8K-shaped calibration set preserve GSM8K
    directions?". Only valid for ``corpus=dolmo``. The GSM8K subdir is
    tiny so ``--nbytes`` is typically not reached — the run consumes the
    whole subdir.
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


# Mix proportions for the official 50B Dolmino mix (Mix % column from the
# dataset card at https://huggingface.co/datasets/allenai/dolmino-mix-1124).
# Each entry is (top-level prefix under data/, target probability). "Stage 2
# Math" in the dataset card aggregates every subdir under data/math/.
DOLMINO_50B_MIX: list[tuple[str, str, float]] = [
    ("dclm",          "data/dclm",          0.472),
    ("flan",          "data/flan",          0.166),
    ("pes2o",         "data/pes2o",         0.0585),
    ("wiki",          "data/wiki",          0.0711),
    ("stackexchange", "data/stackexchange", 0.0245),
    ("math",          "data/math",          0.208),
]


def _list_data_files(repo_id: str) -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return [
        f for f in files
        if f.startswith("data/") and (".json" in f) and not f.endswith("/")
    ]


def _features_for(corpus: str) -> Optional[Features]:
    # Used only as a fallback for shards that need a forced schema. We don't
    # pin features at load-time anymore because Dolmino subdirs have wildly
    # heterogeneous extra columns (DCLM has attributes.*, FLAN has
    # metadata.* etc.) and pinning {"text": "string"} causes the JSON
    # loader to reject any row with additional columns. Instead, the
    # post-load .select_columns(["text"]) in _streaming_ds projects each
    # stream down to a common {text: string} schema, which is what
    # interleave_datasets needs to align.
    return None


def _streaming_ds(repo_id: str, files: Iterable[str], features: Optional[Features]):
    urls = [f"hf://datasets/{repo_id}/{f}" for f in files]
    ds = load_dataset(
        "json",
        data_files={"train": urls},
        split="train",
        streaming=True,
        features=features,
    )
    # Project to the minimal common schema so interleave_datasets can align
    # streams whose source shards have heterogeneous extra columns. We only
    # ever read `text`.
    return ds.select_columns(["text"])


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

    if mix_strategy == "gsm8k":
        if corpus != "dolmo":
            raise ValueError(
                "mix_strategy='gsm8k' is only defined for corpus='dolmo'."
            )
        prefix = "data/math/gsm8k/"
        gsm_files = sorted(f for f in files if f.startswith(prefix))
        if not gsm_files:
            raise SystemExit(
                f"No GSM8K shards found under {prefix} in {repo_id}."
            )
        rng = random.Random(seed)
        rng.shuffle(gsm_files)
        return _streaming_ds(repo_id, gsm_files, features)

    if mix_strategy == "dolmino_50B":
        if corpus != "dolmo":
            raise ValueError(
                "mix_strategy='dolmino_50B' is only defined for corpus='dolmo'."
            )

        # Bucket files by which 50B-mix category their path prefix matches.
        by_category: dict[str, list[str]] = {}
        for f in files:
            for cat, prefix, _ in DOLMINO_50B_MIX:
                if f.startswith(prefix + "/"):
                    by_category.setdefault(cat, []).append(f)
                    break

        rng = random.Random(seed)
        streams = []
        weights = []
        for cat, prefix, weight in DOLMINO_50B_MIX:
            cat_files = sorted(by_category.get(cat, []))
            if not cat_files:
                print(
                    f"[calibration] WARNING: no shards under {prefix}/ for "
                    f"category {cat!r}; skipping (was {weight:.3%} of mix)"
                )
                continue
            rng.shuffle(cat_files)
            streams.append(_streaming_ds(repo_id, cat_files, features))
            weights.append(weight)

        if len(streams) == 1:
            return streams[0]

        # Renormalise in case any category was missing on disk.
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        return interleave_datasets(
            streams,
            probabilities=weights,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    raise ValueError(
        f"Unknown mix_strategy: {mix_strategy!r}. "
        "Expected one of {'first_n', 'shuffle', 'interleave', 'dolmino_50B', "
        "'gsm8k'}."
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
