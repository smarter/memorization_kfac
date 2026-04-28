#!/usr/bin/env python
"""
Wanda baseline (Sun et al. 2024) as K-FAC factors.

Wanda's per-output-row importance is $|W_{oi}| \\cdot \\lVert X_{:,i} \\rVert_2$.
In the K-FAC eigenbasis pipeline this corresponds to:
  - Q_A = Q_G = I (no rotation; natural basis)
  - eva_A[i] proportional to (X^T X)_{ii}             (per-input-dim activation power)
  - eva_G[o] = 1                                       (no gradient information)
  - lambda_correction[o,i] = 1                         (so EKFAC == default importance)

With --weight-coefficients enabled downstream, the resulting importance is
eva_G[o] * eva_A[i] * C_{oi}^2 = (X^T X)_{ii} * W_{oi}^2, which matches
Wanda's ranking (modulo a global per-layer constant from total_processed
that doesn't affect the ranking).

We don't re-run any model forward passes; the diagonal of A^T A has already
been collected by the bergson method. This script just reads the bergson
activation_covariance_sharded files and rewrites them in the on-disk layout
that eval_mem_kfac and mask_intersection expect.
"""
import argparse
import json
import pathlib
import shutil
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_full_per_key(source_inf: pathlib.Path, sub: str) -> Dict[str, torch.Tensor]:
    """Reconstruct full per-key matrices by concatenating shards along dim 0."""
    files = sorted((source_inf / sub).glob("shard_*.safetensors"))
    if not files:
        raise SystemExit(f"No shards found in {source_inf / sub}")
    per_shard = []
    keys = None
    for f in files:
        with safe_open(f, framework="pt") as h:
            d = {k: h.get_tensor(k) for k in h.keys()}
        if keys is None:
            keys = list(d)
        per_shard.append(d)
    return {k: torch.cat([s[k] for s in per_shard], dim=0) for k in keys}


def parse():
    p = argparse.ArgumentParser(description="Derive Wanda factors from a bergson factors dir.")
    p.add_argument("--model", required=True, help="Unused; kept for CLI compatibility.")
    p.add_argument("--save_dir", type=pathlib.Path, required=True)
    p.add_argument(
        "--source",
        type=pathlib.Path,
        required=True,
        help="Path to a bergson factors directory containing influence_results/.",
    )
    p.add_argument("--target_blocks", type=int, nargs="+", required=True)
    # Accept-and-ignore args so this is a drop-in replacement for the bergson
    # collector under the shared ${collect_kfac.args} expansion.
    for opt in ("--device", "--dtype", "--method", "--corpus"):
        p.add_argument(opt, default=None)
    for opt in (
        "--nbytes",
        "--seq_len",
        "--batch_size",
        "--layers_per_pass",
        "--seed",
        "--world_size",
    ):
        p.add_argument(opt, type=int, default=None)
    p.add_argument("--sample_labels", action="store_true")
    p.add_argument("--fsdp", action="store_true")
    return p.parse_args()


def main():
    args = parse()

    src_inf = args.source / "influence_results"
    if not src_inf.exists():
        raise SystemExit(f"Source influence_results not found: {src_inf}")

    A_cov = load_full_per_key(src_inf, "activation_covariance_sharded")
    G_cov = load_full_per_key(src_inf, "gradient_covariance_sharded")

    def in_targets(name: str) -> bool:
        return any(name.startswith(f"layers.{blk}.") for blk in args.target_blocks)

    layer_names = sorted(k for k in A_cov if in_targets(k))
    if not layer_names:
        raise SystemExit(
            f"No layers match target_blocks={args.target_blocks}. "
            f"Source has: {sorted(A_cov.keys())}"
        )

    tp_path = src_inf / "total_processed_covariances.pt"
    total_processed = (
        torch.load(tp_path, weights_only=False) if tp_path.exists() else torch.tensor(1.0)
    )
    tp_f = float(total_processed.item() if hasattr(total_processed, "item") else total_processed)

    # Wanda factors: diagonal A_cov, identity G_cov (scaled so eva_G == 1
    # after eval_mem_kfac normalises by total_processed), identity eigenvectors,
    # and a constant lambda_correction so EKFAC importance matches the
    # default importance (= eva_G ⊗ eva_A) before --weight-coefficients.
    A_payload = {k: torch.diag(A_cov[k].diagonal().float()) for k in layer_names}
    G_payload = {k: tp_f * torch.eye(G_cov[k].shape[-1]) for k in layer_names}
    QA_payload = {k: torch.eye(A_cov[k].shape[-1]) for k in layer_names}
    QG_payload = {k: torch.eye(G_cov[k].shape[-1]) for k in layer_names}
    L_payload = {
        k: tp_f * (A_cov[k].diagonal().float()).repeat(G_cov[k].shape[-1], 1)
        for k in layer_names
    }
    # ^ lambda_correction[o,i] = total_processed * eva_A[i]  →  after dividing
    #   by total_processed_lambda, we get eva_A[i], i.e. EKFAC importance equals
    #   eva_G ⊗ eva_A (with eva_G = 1). Same ranking as the default ec=0 path.

    if args.save_dir.exists():
        shutil.rmtree(args.save_dir)
    influence = args.save_dir / "influence_results"

    payloads = {
        "activation_covariance_sharded": A_payload,
        "gradient_covariance_sharded": G_payload,
        "activation_eigen_sharded": QA_payload,
        "gradient_eigen_sharded": QG_payload,
        "eigenvalue_correction_sharded": L_payload,
    }
    for sub, payload in payloads.items():
        d = influence / sub
        d.mkdir(parents=True, exist_ok=True)
        save_file(payload, d / "shard_0.safetensors")

    torch.save(total_processed, influence / "total_processed_covariances.pt")
    torch.save(total_processed, influence / "total_processed_lambda_correction.pt")

    metadata = {
        "target_modules": layer_names,
        "blocks": sorted(args.target_blocks),
        "format": "bergson",
        "method": "wanda",
        "source": str(args.source),
    }
    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Wrote Wanda factors for {len(layer_names)} modules to {args.save_dir} "
        f"(derived from {args.source})"
    )


if __name__ == "__main__":
    main()
