#!/usr/bin/env python3
"""Compute entropy curves for the Dicke state."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from boundnqs import build_half_hamming_state
from boundnqs import subsystem_entropy_from_tensor
from boundnqs import save_npz_and_json
from boundnqs import apply_plot_style, setup_log_x_integer_ticks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--m-max", type=int, default=None, help="Default: n//2 - 3")
    parser.add_argument("--batch-exp", type=int, default=18)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("results/half_hamming_curve"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)

    psi = build_half_hamming_state(args.n, device=args.device, batch_exp=args.batch_exp)
    psi_t = psi.reshape(*([2] * args.n))
    m_max = args.m_max if args.m_max is not None else (args.n // 2 - 3)
    m_vals = np.arange(1, m_max + 1, dtype=int)
    s_vals = np.array(
        [subsystem_entropy_from_tensor(psi_t, list(range(int(m)))) for m in m_vals],
        dtype=float,
    )

    arrays = {"sizes": m_vals, "entropy": s_vals}
    metadata = {
        "experiment": "half_hamming_curve",
        "n_qubits": args.n,
        "m_max": int(m_max),
        "batch_exp": args.batch_exp,
        "device": args.device,
    }
    npz_path, json_path = save_npz_and_json(args.out_prefix, arrays, metadata)
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")

    if args.plot:
        apply_plot_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        setup_log_x_integer_ticks(ax, m_vals, explicit_ticks=[1, 10] if m_vals.max() >= 10 else [1, int(m_vals.max())])
        ax.plot(m_vals, s_vals, marker="o", ms=3.5, lw=1.8, label=f"n={args.n}")
        ax.set_xlabel(r"$|A|$")
        ax.set_ylabel(r"$S_A$")
        ax.grid(alpha=0.25, which="both", axis="both")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

