#!/usr/bin/env python
"""
OBD diagonal-Fisher collection (Le Cun, Denker & Solla, 1989).

Streams the same Dolmino/OLMo calibration corpus as collect_kfac_bergson.py
but uses bergson's `DiagonalFisherCollector` (HessianConfig.method="obd")
to accumulate the per-entry diagonal of the Fisher information matrix
under Gauss-Newton:

    D[o, i] = sum_n g_{n,o}^2 * a_{n,i}^2.

This is the OBD saliency proxy — the per-entry refinement of Wanda
(adds gradient information per output position) and the per-entry analogue
of K-FAC (drops the Kronecker factorisation, retains cross-sample
correlation between activations and gradients).

To keep the existing eval / mask_intersection / band_activations pipeline
working unchanged, we wrap D as bergson-compatible "K-FAC factors" with
trivial rotations and covariances:

    Q_A = Q_G = I              (no rotation)
    cov_A = total_processed * I  →  eva_A = 1
    cov_G = total_processed * I  →  eva_G = 1
    lambda_correction = D       (so EKFAC importance = D)

With --eigenvalue-corrections --weight-coefficients downstream, the
importance becomes D * W^2, i.e. exactly the OBD saliency S = h_kk * w_k^2.
"""
import argparse
import json
import os
import pathlib
import random
import shutil
from datetime import timedelta
from typing import Dict

import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from bergson.config import DistributedConfig, HessianConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.distributed import launch_distributed_run
from bergson.hessians.hessian_approximations import collect_hessians
from bergson.utils.worker_utils import setup_model_and_peft

# Sibling module; factors out the corpus-mix logic shared with
# collect_kfac_bergson.py.
from calibration import collect_sequences


def parse():
    p = argparse.ArgumentParser(description="Collect OBD diagonal Fisher via bergson.")
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    # Accept-and-ignore --method so this is a drop-in replacement for the
    # bergson collector under the shared ${collect_kfac.args} expansion.
    p.add_argument("--method", default="obd")

    p.add_argument("--corpus", choices=["olmo", "dolmo"], default="dolmo")
    p.add_argument("--nbytes", type=int, default=100_000_000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--calibration_mix",
        choices=["first_n", "shuffle", "interleave"],
        default="first_n",
        help="Corpus shard ordering — see data/calibration.py.",
    )

    p.add_argument("--layers_per_pass", type=int, default=1)
    p.add_argument(
        "--target_blocks",
        type=int,
        nargs="+",
        default=list(range(16)),
        help="0-based decoder block ids",
    )
    p.add_argument("--save_dir", type=pathlib.Path, default="obd_out")
    p.add_argument("--sample_labels", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--world_size", type=int, default=None)
    p.add_argument("--fsdp", action="store_true")
    return p.parse_args()


def obd_worker(
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    ds: Dataset,
    target_modules: set,
):
    """Single-pass OBD worker — no eigendecomposition, no EKFAC pass."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, _ = setup_model_and_peft(index_cfg)

    partial_dir = pathlib.Path(str(index_cfg.partial_run_path))
    total_processed_path = partial_dir / "total_processed.pt"
    cov_total_path = partial_dir / "total_processed_covariances.pt"
    lambda_total_path = partial_dir / "total_processed_lambda_correction.pt"

    kwargs = {
        "model": model,
        "data": ds,
        "index_cfg": index_cfg,
        "hessian_cfg": hessian_cfg,
        "target_modules": target_modules,
        "attention_cfgs": {},
    }
    batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
    kwargs["batches"] = batches

    # Single OBD pass — populates diagonal_fisher_sharded/.
    collect_hessians(**kwargs)
    dist.barrier() if dist.is_initialized() else None

    if rank == 0 and total_processed_path.exists():
        # Eval pipeline reads both names; same scalar.
        shutil.copyfile(total_processed_path, cov_total_path)
        shutil.move(str(total_processed_path), str(lambda_total_path))


def _load_full_per_key(shard_dir: pathlib.Path) -> Dict[str, torch.Tensor]:
    """Reconstruct full per-key matrices by concatenating shards along dim 0."""
    files = sorted(shard_dir.glob("shard_*.safetensors"))
    if not files:
        raise SystemExit(f"No shards found in {shard_dir}")
    keys = None
    per_shard = []
    for f in files:
        with safe_open(f, framework="pt") as h:
            d = {k: h.get_tensor(k) for k in h.keys()}
        if keys is None:
            keys = list(d)
        per_shard.append(d)
    return {k: torch.cat([s[k] for s in per_shard], dim=0) for k in keys}


def synthesize_kfac_compatible_outputs(
    influence_dir: pathlib.Path,
    layer_names: list[str],
    weight_shapes: dict[str, tuple[int, int]],
) -> None:
    """Wrap OBD diag-Fisher as bergson-compatible K-FAC factors.

    Writes identity rotations and constant-scaled covariances so the eval
    pipeline's Rayleigh-quotient eigenvalue derivation gives uniform
    eva_A = eva_G = 1, plus copies the diagonal Fisher into
    eigenvalue_correction_sharded so EKFAC importance = D.
    """
    diag_dir = influence_dir / "diagonal_fisher_sharded"
    D_full = _load_full_per_key(diag_dir)

    tp_path = influence_dir / "total_processed_covariances.pt"
    total_processed = torch.load(tp_path, weights_only=False)
    tp_f = float(total_processed.item() if hasattr(total_processed, "item") else total_processed)

    A_payload, G_payload = {}, {}
    QA_payload, QG_payload = {}, {}
    L_payload = {}

    for name in layer_names:
        if name not in D_full:
            raise SystemExit(
                f"Layer {name} not in OBD diagonal Fisher; available: {sorted(D_full)}"
            )
        out_dim, in_dim = weight_shapes[name]
        A_payload[name] = tp_f * torch.eye(in_dim)
        G_payload[name] = tp_f * torch.eye(out_dim)
        QA_payload[name] = torch.eye(in_dim)
        QG_payload[name] = torch.eye(out_dim)
        L_payload[name] = D_full[name].float()

    payloads = {
        "activation_covariance_sharded": A_payload,
        "gradient_covariance_sharded": G_payload,
        "activation_eigen_sharded": QA_payload,
        "gradient_eigen_sharded": QG_payload,
        "eigenvalue_correction_sharded": L_payload,
    }
    for sub, payload in payloads.items():
        d = influence_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        save_file(payload, d / "shard_0.safetensors")


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

    hf_dataset = Dataset.from_dict(
        {
            "input_ids": sequences,
            "length": [args.seq_len] * len(sequences),
        }
    )

    precision_map = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
    run_path = args.save_dir / "influence_results"

    dist_cfg = DistributedConfig(nproc_per_node=world_size, nnode=1, node_rank=0)

    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=args.model,
        fsdp=args.fsdp,
        precision=precision_map.get(args.dtype, "bf16"),
        token_batch_size=args.batch_size * args.seq_len,
        distributed=dist_cfg,
        overwrite=True,
        gradient_checkpointing=True,
    )

    hessian_cfg = HessianConfig(
        method="obd",
        ev_correction=False,  # OBD is a single pass; no EKFAC.
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
        "obd",
        obd_worker,
        [index_cfg, hessian_cfg, hf_dataset, target_modules],
        dist_cfg,
    )

    if partial.exists():
        if run_path.exists():
            shutil.rmtree(run_path)
        shutil.move(str(partial), str(run_path))

    # Wrap D as bergson-compatible K-FAC factors so downstream stages don't
    # need to know about OBD.
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
    weight_shapes: dict[str, tuple[int, int]] = {}
    for name in target_modules:
        if name.endswith(".gate_proj") or name.endswith(".up_proj"):
            weight_shapes[name] = (intermediate, hidden)
        elif name.endswith(".down_proj"):
            weight_shapes[name] = (hidden, intermediate)
        else:
            raise SystemExit(f"Unexpected target module: {name}")

    print(f"Synthesizing K-FAC-compatible outputs for {len(target_modules)} modules")
    synthesize_kfac_compatible_outputs(run_path, sorted(target_modules), weight_shapes)

    metadata = {
        "target_modules": sorted(target_modules),
        "blocks": sorted(args.target_blocks),
        "format": "bergson",
        "method": "obd",
        "kind": "obd_diag_fisher",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "world_size": world_size,
        "fsdp": args.fsdp,
    }
    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"Saved OBD diagonal Fisher for blocks {args.target_blocks}")


if __name__ == "__main__":
    main()
