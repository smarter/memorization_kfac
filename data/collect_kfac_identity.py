#!/usr/bin/env python
"""
Identity Hessian baseline.

Writes identity covariance / eigenvector matrices in the bergson-compatible
on-disk layout so the rest of the pipeline (eval_mem_kfac, plot_figure2)
treats this like any other K-FAC method but with a trivial curvature
estimate. Useful as a sanity baseline: with --weight-coefficients, the
identity Hessian degenerates to magnitude-based weight pruning of the
selected pairs.

The bergson collection step is skipped entirely; we only need the model
config to derive layer dimensions, so this runs in a few seconds without
GPUs or data.
"""
import argparse
import json
import pathlib
import shutil

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def parse():
    p = argparse.ArgumentParser(description="Identity Hessian baseline")
    p.add_argument("--model", required=True)
    p.add_argument("--save_dir", type=pathlib.Path, required=True)
    p.add_argument(
        "--target_blocks",
        type=int,
        nargs="+",
        required=True,
        help="0-based decoder block indices for which to emit identity factors.",
    )
    # Accept-and-ignore args so this script is a drop-in replacement for
    # collect_kfac_bergson.py under the shared ${collect_kfac.args} expansion.
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--method", default="identity")
    p.add_argument("--corpus", default=None)
    p.add_argument("--nbytes", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--layers_per_pass", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sample_labels", action="store_true")
    p.add_argument("--world_size", type=int, default=None)
    p.add_argument("--fsdp", action="store_true")
    return p.parse_args()


def main():
    args = parse()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size

    # name -> (in_dim, out_dim) following weight shape [out, in].
    shapes: dict[str, tuple[int, int]] = {}
    for blk in args.target_blocks:
        shapes[f"layers.{blk}.mlp.gate_proj"] = (hidden, intermediate)
        shapes[f"layers.{blk}.mlp.up_proj"] = (hidden, intermediate)
        shapes[f"layers.{blk}.mlp.down_proj"] = (intermediate, hidden)

    save_dir: pathlib.Path = args.save_dir
    if save_dir.exists():
        shutil.rmtree(save_dir)
    influence = save_dir / "influence_results"

    sub_payloads: dict[str, dict[str, torch.Tensor]] = {
        "activation_covariance_sharded": {},
        "gradient_covariance_sharded": {},
        "activation_eigen_sharded": {},
        "gradient_eigen_sharded": {},
        "eigenvalue_correction_sharded": {},
    }

    for name, (in_dim, out_dim) in shapes.items():
        sub_payloads["activation_covariance_sharded"][name] = torch.eye(in_dim)
        sub_payloads["gradient_covariance_sharded"][name] = torch.eye(out_dim)
        sub_payloads["activation_eigen_sharded"][name] = torch.eye(in_dim)
        sub_payloads["gradient_eigen_sharded"][name] = torch.eye(out_dim)
        # Lambda[o,i] = eva_G[o] * eva_A[i] = 1 for the identity Hessian.
        sub_payloads["eigenvalue_correction_sharded"][name] = torch.ones(
            out_dim, in_dim
        )

    for sub, payload in sub_payloads.items():
        d = influence / sub
        d.mkdir(parents=True, exist_ok=True)
        save_file(payload, d / "shard_0.safetensors")

    # Downstream eval divides stored values by these scalars.
    torch.save(torch.tensor(1.0), influence / "total_processed_covariances.pt")
    torch.save(torch.tensor(1.0), influence / "total_processed_lambda_correction.pt")

    metadata = {
        "target_modules": sorted(shapes.keys()),
        "blocks": sorted(args.target_blocks),
        "format": "bergson",
        "has_eigendecomposition": True,
        "method": "identity",
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Wrote identity Hessian factors for blocks {sorted(args.target_blocks)} to "
        f"{save_dir}"
    )


if __name__ == "__main__":
    main()
