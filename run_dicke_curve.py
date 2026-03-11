#!/usr/bin/env python3
"""Compute exact Dicke-state entropy curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boundnqs import dicke_entropy_nats
from boundnqs import save_npz_and_json
from boundnqs import apply_plot_style, setup_log_x_integer_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--m-min", type=int, default=2)
    parser.add_argument("--m-max", type=int, default=10)
    parser.add_argument("--excitations", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/dicke_curve"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m_vals = np.arange(args.m_min, args.m_max + 1, dtype=int)
    s_vals = np.array(
        [dicke_entropy_nats(args.n, int(m), excitations=args.excitations) for m in m_vals],
        dtype=float,
    )

    arrays = {"sizes": m_vals, "entropy": s_vals}
    metadata = {
        "experiment": "dicke_curve",
        "n_qubits": args.n,
        "m_min": args.m_min,
        "m_max": args.m_max,
        "excitations": args.excitations,
    }
    npz_path, json_path = save_npz_and_json(args.out_prefix, arrays, metadata)
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")

    if args.plot:
        apply_plot_style()
        fig, ax = plt.subplots(figsize=(2.5, 1.8))
        setup_log_x_integer_ticks(ax, m_vals, explicit_ticks=list(range(args.m_min, args.m_max + 1)))
        ax.plot(m_vals, s_vals, "-o", color="k", ms=3.5, lw=1.5)
        ax.set_xlabel(r"$|A|$")
        ax.set_ylabel(r"$S_A$")
        ax.grid(alpha=0.25, which="both", axis="y")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

