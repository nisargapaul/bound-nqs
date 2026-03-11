#!/usr/bin/env python3
"""Run MLP NQS entropy curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from boundnqs import run_mlp_curve_trials
from boundnqs import save_curve_dict
from boundnqs import apply_plot_style, setup_log_x_integer_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--activations", type=str, default="tanh,sin")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--translations", type=int, default=5)
    parser.add_argument("--batch-exp", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument("--weight-std", type=float, default=1.0)
    parser.add_argument("--bias-std", type=float, default=0.2)
    parser.add_argument("--phase-magnitude", action="store_true")
    parser.add_argument("--alpha-amp", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/mlp_curve"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    activations = [x.strip() for x in args.activations.split(",") if x.strip()]
    curves = run_mlp_curve_trials(
        n_qubits=args.n,
        width=args.width,
        depth=args.depth,
        activations=activations,
        n_trials=args.trials,
        n_translations=args.translations,
        weight_std=args.weight_std,
        bias_std=args.bias_std,
        phase_magnitude=args.phase_magnitude,
        alpha_amp=args.alpha_amp,
        seed=args.seed,
        device=args.device,
        batch_exp=args.batch_exp,
    )

    metadata = {
        "experiment": "mlp_curve",
        "n_qubits": args.n,
        "width": args.width,
        "depth": args.depth,
        "activations": activations,
        "n_trials": args.trials,
        "n_translations": args.translations,
        "batch_exp": args.batch_exp,
        "seed": args.seed,
        "weight_std": args.weight_std,
        "bias_std": args.bias_std,
        "phase_magnitude": args.phase_magnitude,
        "alpha_amp": args.alpha_amp,
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
        palette = {"tanh": "C0", "sin": "C1", "gelu": "C2", "relu": "C3"}
        markers = {"tanh": "o", "sin": "s", "gelu": "D", "relu": "^"}
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
        ax.set_title(f"MLP, w={args.width}, d={args.depth}")
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

