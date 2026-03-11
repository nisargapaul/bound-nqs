#!/usr/bin/env python3
"""Run CosNet/TanhNet-style random-feature experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from boundnqs import page_entropy_nats
from boundnqs import run_random_feature_curve_trials, run_random_feature_hidden_sweep
from boundnqs import save_curve_dict, save_npz_and_json
from boundnqs import apply_plot_style, setup_log_x_integer_ticks


def parse_hidden_sizes(text: str) -> list[int]:
    if text.startswith("geom:"):
        _, lo, hi, count = text.split(":")
        vals = np.unique(np.round(np.geomspace(float(lo), float(hi), int(count))).astype(int))
        return [int(v) for v in vals]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="1,2,5,100",
        help="Comma list or geom:lo:hi:count (e.g. geom:1:100:10).",
    )
    parser.add_argument("--activation", type=str, default="cos", choices=["cos", "tanh"])
    parser.add_argument("--sigma-a", type=float, default=1.0)
    parser.add_argument("--sigma-w", type=float, default=10.0)
    parser.add_argument("--bias-distribution", type=str, default="uniform_pi")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--translations", type=int, default=5)
    parser.add_argument("--batch-exp", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--subsystem-size", type=int, default=None, help="Fixed |A| for hidden sweep.")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["curve", "hidden-sweep", "both"],
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/random_feature"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    apply_plot_style()

    if args.mode in {"curve", "both"}:
        curves = run_random_feature_curve_trials(
            n_qubits=args.n,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            n_trials=args.trials,
            n_translations=args.translations,
            sigma_a=args.sigma_a,
            sigma_w=args.sigma_w,
            bias_distribution=args.bias_distribution,
            seed=args.seed,
            device=args.device,
            batch_exp=args.batch_exp,
        )
        curve_meta = {
            "experiment": "random_feature_curve",
            "n_qubits": args.n,
            "hidden_sizes": hidden_sizes,
            "activation": args.activation,
            "sigma_a": args.sigma_a,
            "sigma_w": args.sigma_w,
            "bias_distribution": args.bias_distribution,
            "n_trials": args.trials,
            "n_translations": args.translations,
            "batch_exp": args.batch_exp,
            "seed": args.seed,
            "device": args.device,
        }
        npz_path, json_path = save_curve_dict(f"{args.out_prefix}_curve", curves, curve_meta)
        print(f"Saved {npz_path}")
        print(f"Saved {json_path}")

        if args.plot:
            fig, ax = plt.subplots(figsize=(6.0, 3.0))
            sizes = next(iter(curves.values())).sizes
            setup_log_x_integer_ticks(ax, sizes)
            cmap = plt.colormaps["viridis"]
            hidden_sorted = sorted(hidden_sizes)
            den = max(1, len(hidden_sorted))
            palette = {h: cmap(i / den) for i, h in enumerate(hidden_sorted)}
            marker_seq = ["o", "s", "^", "D", "v", "P", "X"]
            markers = {h: marker_seq[i % len(marker_seq)] for i, h in enumerate(hidden_sizes)}
            for hidden in hidden_sizes:
                stats = curves[hidden]
                color = palette[hidden]
                ax.fill_between(
                    stats.sizes,
                    np.clip(stats.mean - stats.std, 0.0, None),
                    stats.mean + stats.std,
                    alpha=0.2,
                    linewidth=0,
                    color=color,
                )
                ax.plot(stats.sizes, stats.mean, marker=markers[hidden], ms=4.0, lw=1.8, color=color, label=str(hidden))
            page_vals = np.array([page_entropy_nats(int(m), args.n - int(m)) for m in sizes], dtype=float)
            ax.plot(sizes, page_vals, "k--", lw=1.8, label="Page")
            ax.set_xlabel(r"$|A|$")
            ax.set_ylabel(r"$S_A$")
            ax.set_title(f"{args.activation.capitalize()}Net")
            ax.grid(alpha=0.25, which="both", axis="y")
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                title="# hidden units",
                handlelength=2.6,
            )
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.show()

    if args.mode in {"hidden-sweep", "both"}:
        sweep = run_random_feature_hidden_sweep(
            n_qubits=args.n,
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            subsystem_size=args.subsystem_size,
            n_trials=args.trials,
            n_translations=args.translations,
            sigma_a=args.sigma_a,
            sigma_w=args.sigma_w,
            bias_distribution=args.bias_distribution,
            seed=args.seed,
            device=args.device,
            batch_exp=args.batch_exp,
        )
        sweep_meta = {
            "experiment": "random_feature_hidden_sweep",
            "n_qubits": args.n,
            "activation": args.activation,
            "sigma_a": args.sigma_a,
            "sigma_w": args.sigma_w,
            "bias_distribution": args.bias_distribution,
            "n_trials": args.trials,
            "n_translations": args.translations,
            "batch_exp": args.batch_exp,
            "seed": args.seed,
            "device": args.device,
        }
        arrays = {
            "hidden_sizes": sweep["hidden_sizes"],
            "subsystem_size": np.array([sweep["subsystem_size"]], dtype=int),
            "mean": sweep["mean"],
            "std": sweep["std"],
        }
        npz_path, json_path = save_npz_and_json(f"{args.out_prefix}_hidden_sweep", arrays, sweep_meta)
        print(f"Saved {npz_path}")
        print(f"Saved {json_path}")

        if args.plot:
            fig, ax = plt.subplots(figsize=(4.0, 3.0))
            hidden = sweep["hidden_sizes"]
            mu = sweep["mean"]
            sd = sweep["std"]
            ax.set_xscale("log")
            color = plt.colormaps["plasma"](0.65)
            ax.fill_between(hidden, np.clip(mu - sd, 0.0, None), mu + sd, color=color, alpha=0.2, linewidth=0)
            ax.plot(hidden, mu, marker="o", ms=4.0, lw=1.8, color=color)
            page = page_entropy_nats(int(sweep["subsystem_size"]), args.n - int(sweep["subsystem_size"]))
            ax.axhline(page, color="k", ls="--", lw=1.8, label="Page")
            ax.set_xlabel("# hidden units")
            ax.set_ylabel(r"$S_A$")
            ax.set_title(args.activation.capitalize() + "Net")
            ax.legend(frameon=False, loc="best")
            ax.grid(alpha=0.25, axis="y")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()

