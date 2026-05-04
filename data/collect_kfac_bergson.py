#!/usr/bin/env python
"""
K-FAC factor collection using bergson's EKFAC implementation.

This script provides the same CLI interface as collect_kfac_multilayer.py
but uses bergson's covariance / eigendecomposition / eigenvalue-correction
pipeline with proper valid_mask handling.

It additionally supports multiple GPUs via the --world_size and --fsdp flags.
"""
import argparse
import json
import pathlib
import random
import shutil

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from bergson.config import DistributedConfig, HessianConfig, IndexConfig
from bergson.distributed import launch_distributed_run
from bergson.hessians.hessian_approximations import hessian_worker

# Sibling module; factors out the corpus-mix logic shared with
# collect_kfac_obd.py.
from calibration import collect_sequences


LEGACY_RENAMES = {
    "activation_sharded": "activation_covariance_sharded",
    "gradient_sharded": "gradient_covariance_sharded",
    "eigen_activation_sharded": "activation_eigen_sharded",
    "eigen_gradient_sharded": "gradient_eigen_sharded",
}


def parse():
    """Parse command-line arguments (compatible with collect_kfac_multilayer.py)."""
    p = argparse.ArgumentParser(description="Collect K-FAC factors using bergson")
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument(
        "--method",
        default="kfac",
        choices=["kfac", "tkfac", "shampoo", "foof"],
        help="Hessian approximation method.",
    )

    p.add_argument("--corpus", choices=["olmo", "dolmo"], default="dolmo")
    p.add_argument("--nbytes", type=int, default=100_000_000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--calibration_mix",
        choices=["first_n", "shuffle", "interleave", "dolmino_50B", "gsm8k"],
        default="first_n",
        help="How to order corpus shards during streaming. 'first_n' (default) "
        "consumes alphabetically-first shards only — biases the calibration "
        "toward whichever subdomain comes first lexicographically. 'shuffle' "
        "randomises the shard order globally; 'interleave' round-robins across "
        "top-level subdomain dirs (e.g. data/dclm vs data/openwebmath); "
        "'dolmino_50B' weighted-interleaves to match the published 50B Dolmino "
        "mix (47.2/16.6/5.85/7.11/2.45/20.8 for DCLM/FLAN/pes2o/Wiki/SE/Math); "
        "'gsm8k' restricts to data/math/gsm8k/ only (Dolmino).",
    )

    p.add_argument(
        "--layers_per_pass",
        type=int,
        default=1,
        help="Process this many target layers before flushing.",
    )
    p.add_argument(
        "--target_blocks",
        type=int,
        nargs="+",
        default=list(range(16)),
        help="0-based encoder block ids",
    )
    p.add_argument("--save_dir", type=pathlib.Path, default="kfac_out")
    p.add_argument(
        "--sample_labels",
        action="store_true",
        help="If set, use multinomial-sampled labels",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of GPUs to use. If None, uses all available GPUs when --fsdp is set, otherwise uses 1.",
    )
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallel (FSDP) for multi-GPU training.",
    )
    return p.parse_args()


def _rename_legacy_paths(root: pathlib.Path) -> None:
    """Rename bergson's upstream directory names to the legacy ones expected
    by the rest of memorization_kfac/ and the companion notebooks."""
    for src_name, dst_name in LEGACY_RENAMES.items():
        src = root / src_name
        dst = root / dst_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)


def _materialise_legacy_total_processed(root: pathlib.Path) -> None:
    """The downstream eval pipeline reads ``total_processed_covariances.pt``
    and (when EC is on) ``total_processed_lambda_correction.pt``. Bergson
    only writes one ``total_processed.pt``; mirror it under both legacy
    names so we don't have to plumb a second file through the worker.
    """
    total_processed_path = root / "total_processed.pt"
    if not total_processed_path.exists():
        return
    for legacy in (
        "total_processed_covariances.pt",
        "total_processed_lambda_correction.pt",
    ):
        target = root / legacy
        if not target.exists():
            shutil.copyfile(total_processed_path, target)


def main():
    args = parse()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.world_size is not None:
        world_size = args.world_size
    elif args.fsdp:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    print(f"Streaming ~{args.nbytes / 1e6:.0f}MB from {args.corpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    sequences = collect_sequences(
        args.corpus,
        tokenizer,
        args.seq_len,
        args.nbytes,
        seed=args.seed,
        mix_strategy=args.calibration_mix,
    )
    print(f"Collected {len(sequences)} sequences")

    hf_dataset = Dataset.from_dict({
        "input_ids": sequences,
        "length": [args.seq_len] * len(sequences),
    })

    precision_map = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}

    run_path = args.save_dir / "influence_results"

    dist_cfg = DistributedConfig(
        nproc_per_node=world_size, nnode=1, node_rank=0
    )

    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=args.model,
        fsdp=args.fsdp,
        precision=precision_map.get(args.dtype, "bf16"),
        token_batch_size=args.batch_size * args.seq_len,
        distributed=dist_cfg,
        overwrite=True,
        gradient_checkpointing=True
    )

    hessian_cfg = HessianConfig(
        method=args.method,
        ev_correction=True,
        # fp32 accumulators keep the bf16 outer-product path numerically stable.
        hessian_dtype="fp32",
        use_dataset_labels=not args.sample_labels,
    )

    target_modules = set()
    for blk in args.target_blocks:
        target_modules.add(f"layers.{blk}.mlp.gate_proj")
        target_modules.add(f"layers.{blk}.mlp.up_proj")
        target_modules.add(f"layers.{blk}.mlp.down_proj")

    partial = pathlib.Path(str(run_path) + ".part")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)

    launch_distributed_run(
        "hessian",
        hessian_worker,
        # bergson.hessians.hessian_approximations.hessian_worker takes:
        #   (rank, local_rank, world_size, index_cfg, hessian_cfg, ds,
        #    do_eigendecomposition=True, target_modules=None)
        [index_cfg, hessian_cfg, hf_dataset, True, target_modules],
        dist_cfg,
    )

    if partial.exists():
        if run_path.exists():
            shutil.rmtree(run_path)
        shutil.move(str(partial), str(run_path))

    _materialise_legacy_total_processed(run_path)
    _rename_legacy_paths(run_path)

    metadata = {
        "target_modules": sorted(target_modules),
        "blocks": sorted(args.target_blocks),
        "format": "bergson",
        "has_eigendecomposition": True,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "world_size": world_size,
        "fsdp": args.fsdp,
    }
    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"Saved KFAC factors for blocks {args.target_blocks}")


if __name__ == "__main__":
    main()
