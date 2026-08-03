#!/usr/bin/env python
"""
Experiment 2 (mask intersection) from notes/identity_pruning_experiments.md.

For each MLP weight matrix targeted by the editing pipeline, compare the
K-FAC pair-selection mask against the magnitude (identity) mask at MATCHED
CARDINALITY, in W-space.

The masks live in different bases:
    K-FAC mask M_kfac is over (gradient-eigenvector, activation-eigenvector)
    pairs; identity mask M_id is over (output, input) entries of W. To compare,
    we project the K-FAC reconstruction W'_kfac = U_G (C ⊙ M_kfac) U_A^T back
    into W-space and pull out its top-|M_kfac| entries by magnitude as the
    "natural-basis effective support" of the K-FAC mask.

The script computes all four importance flavours that ``KFACTreatmentPairwise``
can produce in one pass (skipping any that need data not present on disk):
    ec=0 wc=0  importance = eva_G ⊗ eva_A          (paper default)
    ec=0 wc=1  importance = eva_G ⊗ eva_A · C^2    (OBD in K-FAC eigenbasis)
    ec=1 wc=0  importance = lambda_correction      (EKFAC)
    ec=1 wc=1  importance = lambda_correction · C^2

Per layer it reports cardinality, IoU, cosine similarity, reconstruction
errors, Frobenius retention, and cross-mass terms.
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM

# Re-use the loader from the eval pipeline so we don't drift from its
# eigenvalue-derivation conventions (Rayleigh quotient, descending sort,
# lambda re-indexing).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evaluations.eval_mem_kfac import load_bergson_kfac_info  # noqa: E402


def select_pairs_by_mass(importance: torch.Tensor, rho: float) -> torch.Tensor:
    """Top entries until cumulative mass >= rho * total. Returns boolean mask."""
    flat = importance.flatten()
    sorted_vals, sorted_idx = flat.sort(descending=True)
    csum = sorted_vals.double().cumsum(0)
    target = rho * csum[-1].item()
    k = (
        int(
            torch.searchsorted(
                csum, torch.tensor(target, dtype=torch.float64, device=csum.device)
            ).item()
        )
        + 1
    )
    k = min(k, flat.numel())
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[sorted_idx[:k]] = True
    return mask.reshape(importance.shape)


def select_topk_mask(importance: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k entries by importance. Returns boolean mask."""
    flat = importance.flatten()
    _, idx = flat.topk(min(k, flat.numel()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape(importance.shape)


def per_layer_metrics(
    W: torch.Tensor,
    evc_G: torch.Tensor,
    evc_A: torch.Tensor,
    eva_G: torch.Tensor,
    eva_A: torch.Tensor,
    lambda_correction: torch.Tensor | None,
    rho: float,
    use_eigenvalue_corrections: bool,
    use_weight_coefficients: bool,
) -> Dict:
    O, I = W.shape
    n_total = O * I

    # Coefficients of W in the K-FAC eigenbasis: C[o,i] = u_o^T W v_i.
    C = evc_G.T @ W @ evc_A

    if use_eigenvalue_corrections and lambda_correction is not None:
        importance = lambda_correction.float()
    else:
        importance = eva_G[:, None] * eva_A[None, :]
    if use_weight_coefficients:
        importance = importance * (C**2)

    M_kfac = select_pairs_by_mass(importance, rho)
    n_pairs = int(M_kfac.sum().item())

    # Identity mask: top n_pairs entries of W^2.
    M_id = select_topk_mask(W**2, n_pairs)

    # Reconstructions in W-space.
    W_kfac = evc_G @ (C * M_kfac.float()) @ evc_A.T  # dense
    W_id = W * M_id.float()  # sparse, support = M_id

    # Effective natural-basis support of K-FAC: top |M_kfac| magnitudes of W'_kfac.
    M_kfac_in_W = select_topk_mask(W_kfac.abs(), n_pairs)

    # IoU of the natural-basis effective supports.
    intersect = (M_kfac_in_W & M_id).sum().item()
    union = (M_kfac_in_W | M_id).sum().item()
    iou = intersect / max(union, 1)

    # Cosine similarity of the two reconstructions in W-space.
    denom = (W_kfac.norm() * W_id.norm()).item()
    cos = (W_kfac.flatten() @ W_id.flatten()).item() / denom if denom > 0 else 0.0

    # Reconstruction errors.
    norm_W = W.norm().item()
    rel_err_kfac = (W - W_kfac).norm().item() / norm_W
    rel_err_id = (W - W_id).norm().item() / norm_W

    # Frobenius mass retained.
    frob_W2 = (W**2).sum().item()
    frob_kfac2 = (W_kfac**2).sum().item()
    frob_id2 = (W_id**2).sum().item()
    frob_retention_kfac = frob_kfac2 / frob_W2
    frob_retention_id = frob_id2 / frob_W2

    # Cross-mass: how much of K-FAC's reconstructed mass lands at the
    # natural-basis positions M_id "would have kept"? If the K-FAC
    # rotation lines up with magnitude pruning, this is close to 1.
    kfac_mass_at_id = ((W_kfac**2) * M_id.float()).sum().item() / max(frob_kfac2, 1e-30)

    # And the reverse: of W^2's true magnitude mass, how much sits inside
    # K-FAC's effective natural-basis support? Equivalent to frob_retention_id
    # if M_kfac_in_W happened to coincide with M_id.
    id_mass_at_kfac_topk = ((W**2) * M_kfac_in_W.float()).sum().item() / frob_W2

    return {
        "n_pairs": n_pairs,
        "n_total_pairs": n_total,
        "frac_pairs": n_pairs / n_total,
        "rho": rho,
        "iou_top_k_in_W": iou,
        "cosine_W_kfac_W_id": cos,
        "rel_err_kfac": rel_err_kfac,
        "rel_err_id": rel_err_id,
        "frob_retention_kfac": frob_retention_kfac,
        "frob_retention_id": frob_retention_id,
        "kfac_mass_at_id_positions": kfac_mass_at_id,
        "id_mass_at_kfac_topk_in_W": id_mass_at_kfac_topk,
    }


def summarize(per_layer: Dict[str, Dict]) -> Dict:
    keys = [
        "iou_top_k_in_W",
        "cosine_W_kfac_W_id",
        "rel_err_kfac",
        "rel_err_id",
        "frob_retention_kfac",
        "frob_retention_id",
        "kfac_mass_at_id_positions",
        "id_mass_at_kfac_topk_in_W",
    ]
    n = len(per_layer)
    out: Dict = {
        k: {
            "mean": sum(m[k] for m in per_layer.values()) / n,
            "min": min(m[k] for m in per_layer.values()),
            "max": max(m[k] for m in per_layer.values()),
        }
        for k in keys
    }
    out["total_pairs"] = sum(m["n_pairs"] for m in per_layer.values())
    out["total_entries"] = sum(m["n_total_pairs"] for m in per_layer.values())
    out["overall_frac_pairs"] = out["total_pairs"] / out["total_entries"]
    return out


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", required=True)
    p.add_argument("--kfac_factors", type=pathlib.Path, required=True)
    p.add_argument(
        "--layers_json",
        required=True,
        help='JSON like {"13": {"gate": 0.4, "up": 0.4, "down": 1.0}, ...}; '
        "projections at mass=1.0 are skipped (= no editing target).",
    )
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def collect_layer_specs(layers: Dict) -> List[Tuple[str, float]]:
    """({block: {proj: mass}}) -> list of (full_layer_name, rho), mass<1.0 only."""
    specs: List[Tuple[str, float]] = []
    for blk, projs in layers.items():
        for proj, mass in projs.items():
            mass_f = float(mass)
            if mass_f >= 1.0:
                continue
            specs.append((f"model.layers.{blk}.mlp.{proj}_proj", mass_f))
    return specs


def main():
    args = parse()
    layers = json.loads(args.layers_json)
    layer_specs = collect_layer_specs(layers)
    if not layer_specs:
        raise SystemExit("No layers with mass < 1.0; nothing to compare.")
    layer_names = [n for n, _ in layer_specs]

    print(f"Loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(args.device)

    print(f"Loading K-FAC factors from {args.kfac_factors}")
    info_no_ec = load_bergson_kfac_info(
        bergson_path=str(args.kfac_factors),
        layer_names=layer_names,
        model=model,
        device=args.device,
        use_eigenvalue_corrections=False,
    )

    try:
        info_ec = load_bergson_kfac_info(
            bergson_path=str(args.kfac_factors),
            layer_names=layer_names,
            model=model,
            device=args.device,
            use_eigenvalue_corrections=True,
        )
        has_ec = True
    except FileNotFoundError as e:
        print(f"  (no eigenvalue corrections on disk: {e})")
        info_ec = None
        has_ec = False

    combinations: List[Tuple[bool, bool]] = [(False, False), (False, True)]
    if has_ec:
        combinations.extend([(True, False), (True, True)])

    by_setting: Dict[str, Dict] = {}
    for use_ec, use_wc in combinations:
        label = f"ec{int(use_ec)}_wc{int(use_wc)}"
        info = info_ec if use_ec else info_no_ec
        per_layer: Dict[str, Dict] = {}
        for layer_name, rho in layer_specs:
            ld = info[layer_name]
            per_layer[layer_name] = per_layer_metrics(
                W=ld["W_orig"].float(),
                evc_G=ld["evc_G"].float(),
                evc_A=ld["evc_A"].float(),
                eva_G=ld["eva_G"].float(),
                eva_A=ld["eva_A"].float(),
                lambda_correction=ld.get("lambda_correction") if use_ec else None,
                rho=rho,
                use_eigenvalue_corrections=use_ec,
                use_weight_coefficients=use_wc,
            )
        summary = summarize(per_layer)
        by_setting[label] = {
            "use_eigenvalue_corrections": use_ec,
            "use_weight_coefficients": use_wc,
            "per_layer": per_layer,
            "summary": summary,
        }

        print(
            f"\n[{label}] frac_pairs={summary['overall_frac_pairs']:.4%}  "
            f"IoU(mean)={summary['iou_top_k_in_W']['mean']:.4f}  "
            f"cos(mean)={summary['cosine_W_kfac_W_id']['mean']:.4f}  "
            f"err_kfac={summary['rel_err_kfac']['mean']:.4f}  "
            f"err_id={summary['rel_err_id']['mean']:.4f}"
        )
        for layer_name, m in per_layer.items():
            print(
                f"  {layer_name}: n_pairs={m['n_pairs']} "
                f"({m['frac_pairs']:.3%}) "
                f"IoU={m['iou_top_k_in_W']:.3f} "
                f"cos={m['cosine_W_kfac_W_id']:.3f} "
                f"err_kfac={m['rel_err_kfac']:.3f} "
                f"err_id={m['rel_err_id']:.3f}"
            )

    output = {
        "args": {
            "model": args.model,
            "kfac_factors": str(args.kfac_factors),
            "layers": layers,
        },
        "has_eigenvalue_corrections": has_ec,
        "by_setting": by_setting,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
