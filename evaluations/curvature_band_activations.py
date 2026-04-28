#!/usr/bin/env python
"""
Per-task activation-band analysis on K-FAC's activation eigenbasis.

For each MLP layer, hook the input to the chosen projection (default `up`),
project activations onto K-FAC's activation-eigenvector bands (top-10%,
10-25%, 25-50%, bottom-50% by activation eigenvalue), and report the mean
L2 norm of the projection per dataset.

Generalises the paper's Figure 2 (which compares "memorized" vs "clean") to
arbitrary datasets, so we can ask whether GSM8K activations concentrate on
the bottom-of-the-spectrum K-FAC eigenvectors more than Winogrande's do.
That's the smoking-gun signal for "the K-FAC basis is anti-aligned with
math-specialised weights at the calibration distribution it's estimated from."

Output: results.json with per-(layer, band, dataset) strengths plus the
cross-dataset and cross-band ratios you need to read off the hypothesis.
"""
import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use the bergson-format loader from eval_mem_kfac for eigenvector / eigenvalue
# derivation conventions (Rayleigh quotient, descending sort).
from evaluations.eval_mem_kfac import (  # noqa: E402
    _load_bergson_sharded,
    _normalize_bergson_keys,
)

BANDS = {
    "top_10": (0.00, 0.10),
    "10_25": (0.10, 0.25),
    "25_50": (0.25, 0.50),
    "bottom_50": (0.50, 1.00),
}


def load_eigen_for_layer(
    factors_dir: pathlib.Path, bergson_key: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (Q_A_sorted_desc, eva_A_sorted_desc, total_processed) for one layer.

    Q_A's columns are eigenvectors in descending eigenvalue order — matches
    the convention `eval_mem_kfac.load_bergson_kfac_info` uses downstream.
    """
    influence = factors_dir / "influence_results"
    act_eigen = _normalize_bergson_keys(_load_bergson_sharded(influence / "activation_eigen_sharded"))
    act_cov = _normalize_bergson_keys(_load_bergson_sharded(influence / "activation_covariance_sharded"))
    total_processed = torch.load(influence / "total_processed_covariances.pt", weights_only=False)
    tp = total_processed.item() if hasattr(total_processed, "item") else float(total_processed)

    if bergson_key not in act_eigen:
        raise KeyError(f"Layer {bergson_key} not in factors at {influence}")

    Q = act_eigen[bergson_key].float().to(device)
    A = act_cov[bergson_key].float().to(device) / tp
    A = (A + A.T) / 2

    # Rayleigh quotient → eigenvalues; sort descending.
    eva = (Q.T @ A @ Q).diagonal()
    idx = eva.argsort(descending=True)
    return Q[:, idx], eva[idx], tp


def band_eigenvector_slice(
    Q_sorted: torch.Tensor, band_lo: float, band_hi: float
) -> torch.Tensor:
    """Return the columns of Q in the [lo, hi) percentile band by eigenvalue rank."""
    n = Q_sorted.shape[1]
    i_lo = int(round(band_lo * n))
    i_hi = int(round(band_hi * n))
    return Q_sorted[:, i_lo:i_hi]


def load_dataset_prompts(spec: dict, n_samples: int, tokenizer, max_seq_len: int) -> torch.Tensor:
    """Tokenize n_samples prompts from a HuggingFace dataset spec.

    spec keys:
      hf_path:    HF dataset id (e.g. "gsm8k")
      hf_config:  optional sub-config (e.g. "main")
      split:      train/validation/test (default "train")
      column:     which text column to read (default "text")
      template:   optional Python f-string applied to each row dict
                  (e.g. "{question}\\nAnswer:" for GSM8K few-shot)
    Returns: input_ids tensor, shape [n_samples, max_seq_len], left-padded
    to max_seq_len with the eos token if shorter.
    """
    from datasets import load_dataset

    hf_path = spec["hf_path"]
    cfg = spec.get("hf_config")
    split = spec.get("split", "train")
    column = spec.get("column", "text")
    template = spec.get("template")

    ds = load_dataset(hf_path, cfg, split=split, streaming=True)
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id or 0

    rows: List[List[int]] = []
    for row in ds:
        if template is not None:
            text = template.format(**row)
        else:
            text = row[column]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        ids = ids[:max_seq_len]
        if len(ids) < max_seq_len:
            ids = ids + [pad_id] * (max_seq_len - len(ids))
        rows.append(ids)
        if len(rows) >= n_samples:
            break
    if len(rows) < n_samples:
        print(f"  [warn] only {len(rows)} samples available from {hf_path}; padding to {n_samples}")
    return torch.tensor(rows, dtype=torch.long)


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", required=True)
    p.add_argument("--kfac_factors", type=pathlib.Path, required=True)
    p.add_argument(
        "--datasets_json",
        required=True,
        help='JSON dict {name: {hf_path, hf_config, split, column, template}}',
    )
    p.add_argument("--target_blocks", type=int, nargs="+", required=True)
    p.add_argument(
        "--projection",
        choices=["gate", "up", "down"],
        default="up",
        help="Which MLP projection's input to capture.",
    )
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    return p.parse_args()


def make_bergson_key(blk: int, projection: str) -> str:
    return f"layers.{blk}.mlp.{projection}_proj"


def main():
    args = parse()
    datasets_spec = json.loads(args.datasets_json)
    if not datasets_spec:
        raise SystemExit("No datasets specified.")

    print(f"Loading model {args.model}")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()

    # Load eigenvectors and split into bands per target layer.
    layer_band_evecs: Dict[int, Dict[str, torch.Tensor]] = {}
    layer_band_sizes: Dict[int, Dict[str, int]] = {}
    for blk in args.target_blocks:
        key = make_bergson_key(blk, args.projection)
        Q_sorted, eva, _ = load_eigen_for_layer(args.kfac_factors, key, args.device)
        Q_sorted = Q_sorted.to(dtype)
        layer_band_evecs[blk] = {
            band: band_eigenvector_slice(Q_sorted, lo, hi) for band, (lo, hi) in BANDS.items()
        }
        layer_band_sizes[blk] = {
            band: layer_band_evecs[blk][band].shape[1] for band in BANDS
        }
        print(f"  layer {blk}: {eva.shape[0]} eigenvalues, sizes={layer_band_sizes[blk]}")

    # Streaming accumulators: for each (dataset, layer, band):
    #   sum_norm: cumulative L2-norm-per-token of band-projection
    #   sum_norm_sq: cumulative squared L2 norm
    #   tokens: number of tokens accumulated
    sum_norm: Dict[Tuple[str, int, str], float] = collections.defaultdict(float)
    sum_norm_sq: Dict[Tuple[str, int, str], float] = collections.defaultdict(float)
    tokens: Dict[Tuple[str, int], int] = collections.defaultdict(int)

    # Hooks: capture input to mlp.{projection}_proj for each target layer.
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(blk):
        def hook(module, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            captured[blk] = x.detach()
        return hook

    handles = []
    for blk in args.target_blocks:
        proj = getattr(model.model.layers[blk].mlp, f"{args.projection}_proj")
        handles.append(proj.register_forward_pre_hook(make_hook(blk)))

    try:
        for ds_name, spec in datasets_spec.items():
            print(f"\nProcessing dataset {ds_name} ({spec.get('hf_path', '?')})")
            input_ids = load_dataset_prompts(spec, args.n_samples, tokenizer, args.seq_len)
            input_ids = input_ids.to(args.device)
            n = input_ids.shape[0]
            with torch.no_grad():
                for i in range(0, n, args.batch_size):
                    batch = input_ids[i : i + args.batch_size]
                    captured.clear()
                    model(batch)
                    # Activations are now in captured[blk] for each blk.
                    for blk in args.target_blocks:
                        x = captured[blk]  # [B, S, H]
                        x_flat = x.reshape(-1, x.shape[-1])  # [B*S, H]
                        n_tok = x_flat.shape[0]
                        for band, evecs in layer_band_evecs[blk].items():
                            if evecs.shape[1] == 0:
                                continue
                            # [B*S, H] @ [H, n_band] -> [B*S, n_band]
                            proj = (x_flat @ evecs).float()
                            norms = proj.norm(dim=-1)  # [B*S]
                            sum_norm[(ds_name, blk, band)] += norms.sum().item()
                            sum_norm_sq[(ds_name, blk, band)] += (norms * norms).sum().item()
                        tokens[(ds_name, blk)] += n_tok
    finally:
        for h in handles:
            h.remove()

    # Aggregate.
    per_layer: Dict[str, dict] = {}
    for blk in args.target_blocks:
        layer_out: Dict[str, dict] = {"datasets": {}, "ratios": {}}
        for ds_name in datasets_spec:
            n_tok = tokens[(ds_name, blk)]
            ds_strengths: Dict[str, float] = {}
            ds_strengths_sq: Dict[str, float] = {}
            for band in BANDS:
                k = (ds_name, blk, band)
                if n_tok > 0:
                    ds_strengths[band] = sum_norm[k] / n_tok
                    ds_strengths_sq[band] = sum_norm_sq[k] / n_tok
                else:
                    ds_strengths[band] = 0.0
                    ds_strengths_sq[band] = 0.0
            layer_out["datasets"][ds_name] = {
                "mean_l2_norm_per_token": ds_strengths,
                "mean_squared_l2_norm_per_token": ds_strengths_sq,
                "n_tokens": n_tok,
                "band_sizes": layer_band_sizes[blk],
                "bottom_50_over_top_10": (
                    ds_strengths["bottom_50"] / ds_strengths["top_10"]
                    if ds_strengths.get("top_10", 0) > 0 else 0.0
                ),
            }

        # Cross-dataset ratios per band: ds_a / ds_b, for every ordered pair.
        ds_list = list(datasets_spec)
        for ds_a in ds_list:
            for ds_b in ds_list:
                if ds_a == ds_b:
                    continue
                key = f"{ds_a}_over_{ds_b}"
                layer_out["ratios"][key] = {}
                for band in BANDS:
                    a = layer_out["datasets"][ds_a]["mean_l2_norm_per_token"][band]
                    b = layer_out["datasets"][ds_b]["mean_l2_norm_per_token"][band]
                    layer_out["ratios"][key][band] = (a / b) if b > 0 else 0.0
        per_layer[str(blk)] = layer_out

    # Summary: per (ds_a, ds_b) pair, find the layer with the largest
    # bottom_50/top_10 ratio gap — i.e. where ds_a most "concentrates on
    # low-curvature directions relative to ds_b."
    summary: Dict[str, dict] = {"max_bottom_to_top_signature": {}}
    ds_list = list(datasets_spec)
    for ds_a in ds_list:
        for ds_b in ds_list:
            if ds_a == ds_b:
                continue
            key = f"{ds_a}_over_{ds_b}"
            best_blk = None
            best_signature = -float("inf")
            for blk in args.target_blocks:
                lo = per_layer[str(blk)]["ratios"][key]["bottom_50"]
                hi = per_layer[str(blk)]["ratios"][key]["top_10"]
                signature = lo - hi
                if signature > best_signature:
                    best_signature = signature
                    best_blk = blk
            summary["max_bottom_to_top_signature"][key] = {
                "layer": best_blk,
                "ratio_in_bottom_50_minus_ratio_in_top_10": best_signature,
                "ratio_in_bottom_50": per_layer[str(best_blk)]["ratios"][key]["bottom_50"]
                if best_blk is not None
                else 0.0,
                "ratio_in_top_10": per_layer[str(best_blk)]["ratios"][key]["top_10"]
                if best_blk is not None
                else 0.0,
            }
            print(
                f"\n[{key}] strongest layer = {best_blk}: "
                f"bottom_50 ratio = {summary['max_bottom_to_top_signature'][key]['ratio_in_bottom_50']:.3f}, "
                f"top_10 ratio = {summary['max_bottom_to_top_signature'][key]['ratio_in_top_10']:.3f}, "
                f"signature = {best_signature:+.3f}"
            )

    output = {
        "args": {
            "model": args.model,
            "kfac_factors": str(args.kfac_factors),
            "datasets": datasets_spec,
            "target_blocks": sorted(args.target_blocks),
            "projection": args.projection,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        },
        "bands": list(BANDS),
        "per_layer": per_layer,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
