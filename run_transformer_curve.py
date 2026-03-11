#!/usr/bin/env python3
"""Run patch-transformer NQS entropy curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from boundnqs import run_transformer_curve_trials
from boundnqs import save_curve_dict
from boundnqs import apply_plot_style, setup_log_x_integer_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff-ratio", type=float, default=2.0)
    parser.add_argument("--activations", type=str, default="tanh,sin")
    parser.add_argument("--alpha-amp", type=float, default=1.0)
    parser.add_argument("--freeze-random-heads", action="store_true")
    parser.add_argument("--weight-std", type=float, default=1.0)
    parser.add_argument("--bias-std", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--translations", type=int, default=5)
    parser.add_argument("--batch-exp", type=int, default=14)
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/transformer_curve"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    activations = [x.strip() for x in args.activations.split(",") if x.strip()]
    curves = run_transformer_curve_trials(
        n_qubits=args.n,
        patch_size=args.patch_size,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff_ratio=args.d_ff_ratio,
        activations=activations,
        alpha_amp=args.alpha_amp,
        freeze_random_heads=args.freeze_random_heads,
        n_trials=args.trials,
        n_translations=args.translations,
        weight_std=args.weight_std,
        bias_std=args.bias_std,
        seed=args.seed,
        device=args.device,
        batch_exp=args.batch_exp,
    )

    metadata = {
        "experiment": "transformer_curve",
        "n_qubits": args.n,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff_ratio": args.d_ff_ratio,
        "activations": activations,
        "alpha_amp": args.alpha_amp,
        "freeze_random_heads": args.freeze_random_heads,
        "weight_std": args.weight_std,
        "bias_std": args.bias_std,
        "n_trials": args.trials,
        "n_translations": args.translations,
        "batch_exp": args.batch_exp,
        "seed": args.seed,
        "device": args.device,
    }
    npz_path, json_path = save_curve_dict(args.out_prefix, curves, metadata)
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")

    if args.plot:
        apply_plot_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        m_vals = next(iter(curves.values())).sizes
        setup_log_x_integer_ticks(ax, m_vals)
        palette = {"tanh": "C0", "sin": "C1", "softplus": "C2"}
        markers = {"tanh": "o", "sin": "s", "softplus": "D"}
        for name, stats in curves.items():
            color = palette.get(name, None)
            marker = markers.get(name, "o")
            ax.fill_between(
                stats.sizes,
                stats.mean - stats.std,
                stats.mean + stats.std,
                alpha=0.2,
                linewidth=0,
                color=color,
            )
            ax.plot(stats.sizes, stats.mean, marker=marker, ms=4.0, lw=1.8, label=name, color=color)
        ax.set_xlabel(r"$|A|$")
        ax.set_ylabel(r"$S_A$")
        ax.set_title(
            f"Transformer, P={args.patch_size}, s={args.stride}, "
            f"L={args.n_layers}, d={args.d_model}, H={args.n_heads}"
        )
        ax.grid(alpha=0.25, which="both", axis="y")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="nonlinearity",
            handlelength=2.8,
        )
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()


if __name__ == "__main__":
    main()

