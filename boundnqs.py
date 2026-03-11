"""
boundnqs — Code for: Bound on entanglement in neural quantum states.

Nisarga Paul, arXiv:2510.11797
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import ScalarFormatter


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def get_activation(name: str):
    """Return an elementwise activation."""
    key = name.lower()
    if key == "tanh":
        return torch.tanh
    if key == "sin":
        return torch.sin
    if key == "relu":
        return F.relu
    if key == "gelu":
        return F.gelu
    if key == "softplus":
        return F.softplus
    if key == "sigmoid":
        return torch.sigmoid
    if key == "elu":
        return F.elu
    if key == "cos":
        return torch.cos
    raise ValueError(f"Unknown activation: {name}")


def init_linear_normal(layer: nn.Linear, weight_std: float, bias_std: float) -> None:
    """Fan-in-scaled normal initialization."""
    nn.init.normal_(layer.weight, std=weight_std / math.sqrt(layer.in_features))
    if layer.bias is not None:
        nn.init.normal_(layer.bias, std=bias_std)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLPNQS(nn.Module):
    """
    Depth-d, width-w MLP over spin strings in {-1,+1}^N.

    By default returns complex affine head output re + i im.
    Optional phase-magnitude mode returns exp(alpha*re + i im).
    """

    def __init__(
        self,
        n_qubits: int,
        width: int,
        depth: int,
        activation: str = "tanh",
        weight_std: float = 1.0,
        bias_std: float = 0.1,
        phase_magnitude: bool = False,
        alpha_amp: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.width = int(width)
        self.depth = int(depth)
        self.activation = get_activation(activation)
        self.phase_magnitude = bool(phase_magnitude)
        self.alpha_amp = float(alpha_amp)
        self.weight_std = float(weight_std)
        self.bias_std = float(bias_std)

        hidden_layers = []
        in_dim = self.n_qubits
        for _ in range(self.depth):
            layer = nn.Linear(in_dim, self.width, bias=True)
            init_linear_normal(layer, self.weight_std, self.bias_std)
            hidden_layers.append(layer)
            in_dim = self.width
        self.hidden = nn.ModuleList(hidden_layers)

        self.head_re = nn.Linear(self.width, 1, bias=True)
        self.head_im = nn.Linear(self.width, 1, bias=True)
        init_linear_normal(self.head_re, self.weight_std, self.bias_std)
        init_linear_normal(self.head_im, self.weight_std, self.bias_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = self.activation(layer(h))
        re = self.head_re(h).squeeze(-1)
        im = self.head_im(h).squeeze(-1)
        if self.phase_magnitude:
            if self.alpha_amp == 0.0:
                return torch.exp(1j * im)
            return torch.exp(self.alpha_amp * re) * torch.exp(1j * im)
        return torch.complex(re, im)


class PatchTransformerNQS(nn.Module):
    """Patch-token Transformer NQS with optional fixed random output heads."""

    def __init__(
        self,
        n_qubits: int,
        patch_size: int = 3,
        stride: int = 2,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff_ratio: float = 2.0,
        activation: str = "tanh",
        alpha_amp: float = 0.0,
        weight_std: float = 1.0,
        bias_std: float = 0.2,
        freeze_random_heads: bool = False,
    ) -> None:
        super().__init__()
        if patch_size < 1 or stride < 1:
            raise ValueError("patch_size and stride must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_qubits = int(n_qubits)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.n_tokens = (self.n_qubits - self.patch_size) // self.stride + 1
        if self.n_tokens <= 0:
            raise ValueError("Invalid patch configuration for given n_qubits.")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_head = self.d_model // self.n_heads
        self.d_ff = int(d_ff_ratio * self.d_model)
        self.activation = get_activation(activation)
        self.alpha_amp = float(alpha_amp)
        self.weight_std = float(weight_std)
        self.bias_std = float(bias_std)

        self.embed = nn.Linear(self.patch_size, self.d_model, bias=True)
        init_linear_normal(self.embed, self.weight_std, self.bias_std)

        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        self.o_layers = nn.ModuleList()
        self.ff1_layers = nn.ModuleList()
        self.ff2_layers = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.ln2 = nn.ModuleList()

        for _ in range(self.n_layers):
            q = nn.Linear(self.d_model, self.d_model, bias=True)
            k = nn.Linear(self.d_model, self.d_model, bias=True)
            v = nn.Linear(self.d_model, self.d_model, bias=True)
            o = nn.Linear(self.d_model, self.d_model, bias=True)
            ff1 = nn.Linear(self.d_model, self.d_ff, bias=True)
            ff2 = nn.Linear(self.d_ff, self.d_model, bias=True)
            for layer in (q, k, v, o, ff1, ff2):
                init_linear_normal(layer, self.weight_std, self.bias_std)
            self.q_layers.append(q)
            self.k_layers.append(k)
            self.v_layers.append(v)
            self.o_layers.append(o)
            self.ff1_layers.append(ff1)
            self.ff2_layers.append(ff2)
            self.ln1.append(nn.LayerNorm(self.d_model))
            self.ln2.append(nn.LayerNorm(self.d_model))

        output_dim = self.n_tokens * self.d_model
        self.head_re = nn.Linear(output_dim, 1, bias=True)
        self.head_im = nn.Linear(output_dim, 1, bias=True)
        init_linear_normal(self.head_re, self.weight_std, self.bias_std)
        init_linear_normal(self.head_im, self.weight_std, self.bias_std)

        if freeze_random_heads:
            with torch.no_grad():
                head_dtype = self.head_re.weight.dtype
                v_re = torch.randn(output_dim, dtype=head_dtype)
                v_im = torch.randn(output_dim, dtype=head_dtype)
                v_re /= torch.linalg.norm(v_re)
                v_im /= torch.linalg.norm(v_im)
                self.head_re.weight.copy_(v_re.view(1, -1))
                self.head_re.bias.zero_()
                self.head_im.weight.copy_(v_im.view(1, -1))
                self.head_im.bias.zero_()
            for p in list(self.head_re.parameters()) + list(self.head_im.parameters()):
                p.requires_grad = False

    def _patchify(self, x_pm1: torch.Tensor) -> torch.Tensor:
        starts = torch.arange(
            0,
            self.n_qubits - self.patch_size + 1,
            self.stride,
            device=x_pm1.device,
            dtype=torch.long,
        )
        idx = starts.unsqueeze(1) + torch.arange(self.patch_size, device=x_pm1.device).unsqueeze(0)
        return x_pm1[:, idx]

    def _mh_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = q.shape
        q = q.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        z = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return z.view(batch_size, n_tokens, self.d_model)

    def forward(self, x_pm1: torch.Tensor) -> torch.Tensor:
        batch_size, n_qubits = x_pm1.shape
        if n_qubits != self.n_qubits:
            raise ValueError(f"Expected n_qubits={self.n_qubits}, got {n_qubits}.")

        h = self.embed(self._patchify(x_pm1))
        for layer_idx in range(self.n_layers):
            q = self.q_layers[layer_idx](h)
            k = self.k_layers[layer_idx](h)
            v = self.v_layers[layer_idx](h)
            attn_out = self._mh_attention(q, k, v)
            attn_out = self.o_layers[layer_idx](attn_out)
            h = self.ln1[layer_idx](h + attn_out)

            ff = self.ff1_layers[layer_idx](h)
            ff = self.activation(ff)
            ff = self.ff2_layers[layer_idx](ff)
            h = self.ln2[layer_idx](h + ff)

        flat = h.reshape(batch_size, self.n_tokens * self.d_model)
        re = self.head_re(flat).squeeze(-1)
        im = self.head_im(flat).squeeze(-1)
        if self.alpha_amp == 0.0:
            return torch.exp(1j * im)
        return torch.exp(self.alpha_amp * re) * torch.exp(1j * im)


class RandomFeatureNQS(nn.Module):
    """
    One-layer random-feature model used for CosNet/TanhNet style baselines.

    f(s) = sum_i a_i * phi(w_i . s + b_i), with independent real/imag channels.
    """

    def __init__(
        self,
        n_qubits: int,
        hidden_units: int,
        activation: str = "cos",
        sigma_a: float = 1.0,
        sigma_w: float = 10.0,
        bias_distribution: str = "uniform_pi",
    ) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.hidden_units = int(hidden_units)
        self.sigma_a = float(sigma_a)
        self.sigma_w = float(sigma_w)
        self.activation_name = activation.lower()
        self.activation = get_activation(self.activation_name)
        self.bias_distribution = bias_distribution

        scale_w = self.sigma_w / math.sqrt(self.n_qubits)
        scale_a = self.sigma_a / math.sqrt(self.hidden_units)

        self.w_re = nn.Parameter(torch.randn(self.hidden_units, self.n_qubits) * scale_w, requires_grad=False)
        self.w_im = nn.Parameter(torch.randn(self.hidden_units, self.n_qubits) * scale_w, requires_grad=False)
        self.a_re = nn.Parameter(torch.randn(self.hidden_units) * scale_a, requires_grad=False)
        self.a_im = nn.Parameter(torch.randn(self.hidden_units) * scale_a, requires_grad=False)

        self.b_re = nn.Parameter(self._sample_bias(self.hidden_units), requires_grad=False)
        self.b_im = nn.Parameter(self._sample_bias(self.hidden_units), requires_grad=False)

    def _sample_bias(self, size: int) -> torch.Tensor:
        if self.bias_distribution == "uniform_pi":
            return torch.empty(size).uniform_(-math.pi, math.pi)
        if self.bias_distribution == "normal_unit":
            return torch.randn(size)
        raise ValueError(f"Unknown bias_distribution={self.bias_distribution}")

    def _channel(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = x @ w.T + b
        return self.activation(z) @ a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re = self._channel(x, self.w_re, self.b_re, self.a_re)
        im = self._channel(x, self.w_im, self.b_im, self.a_im)
        return torch.complex(re, im)


@dataclass(frozen=True)
class SingleNonlinearityConfig:
    """Configuration for the single-global-nonlinearity ansatz."""

    n_qubits: int
    activation: str
    mode: str = "real"
    use_fan_in: bool = False
    weight_std: float = 1.0
    bias_std: float = 1.0
    coeff_real: float | None = None
    coeff_imag: float | None = None


class SingleNonlinearityNQS(nn.Module):
    """
    psi(s) = exp( (a_real + i a_imag) * sigma((w.s + b)/scale) ).

    mode:
      - "real":      a = 1 + 0i
      - "pure_phase": a = 0 + 1i
      - "general":   a sampled or provided by coeff_real/coeff_imag
    """

    def __init__(self, cfg: SingleNonlinearityConfig):
        super().__init__()
        self.cfg = cfg
        self.n_qubits = int(cfg.n_qubits)
        self.activation = get_activation(cfg.activation)

        self.w = nn.Parameter(torch.randn(self.n_qubits) * float(cfg.weight_std), requires_grad=False)
        self.b = nn.Parameter(torch.randn(()) * float(cfg.bias_std), requires_grad=False)

        if cfg.mode == "real":
            a_real, a_imag = 1.0, 0.0
        elif cfg.mode == "pure_phase":
            a_real, a_imag = 0.0, 1.0
        elif cfg.mode == "general":
            a_real = float(cfg.coeff_real) if cfg.coeff_real is not None else float(torch.randn(()))
            a_imag = float(cfg.coeff_imag) if cfg.coeff_imag is not None else float(torch.randn(()))
        else:
            raise ValueError("mode must be one of: real, pure_phase, general")
        coeff_dtype = self.w.dtype
        self.a_real = nn.Parameter(torch.tensor(a_real, dtype=coeff_dtype), requires_grad=False)
        self.a_imag = nn.Parameter(torch.tensor(a_imag, dtype=coeff_dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x @ self.w + self.b
        if self.cfg.use_fan_in:
            t = t / math.sqrt(self.n_qubits)
        f = self.activation(t)
        z = (self.a_real + 1j * self.a_imag) * f
        return torch.exp(z)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def _pm1_batch_from_ints(
    start: int,
    batch_size: int,
    n_qubits: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert integers in [start, start+batch_size) to {-1,+1} bitstrings."""
    ints = torch.arange(start, start + batch_size, device=device, dtype=torch.int64)
    bit_positions = torch.arange(n_qubits, device=device, dtype=torch.int64)
    bits01 = ((ints.unsqueeze(1) >> bit_positions) & 1).to(dtype)
    bits01 = torch.flip(bits01, dims=[1])
    return bits01 * 2.0 - 1.0


@torch.no_grad()
def build_normalized_state(
    model: torch.nn.Module,
    n_qubits: int | None = None,
    device: torch.device | str | None = None,
    batch_exp: int = 14,
) -> torch.Tensor:
    """
    Build the full normalized statevector psi over all 2^N basis strings.

    The model must accept input shape (batch, n_qubits) with entries in {-1,+1}
    and return complex amplitudes with shape (batch,).
    """
    if n_qubits is None:
        n_qubits = getattr(model, "n_qubits", None)
    if n_qubits is None:
        raise ValueError("n_qubits must be provided if model does not expose n_qubits.")

    if device is None:
        device = next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    try:
        input_dtype = next(model.parameters()).dtype
    except StopIteration:
        input_dtype = torch.float64

    total_states = 1 << int(n_qubits)
    batch_size = 1 << min(int(n_qubits), int(batch_exp))
    chunks: list[torch.Tensor] = []
    for start in range(0, total_states, batch_size):
        cur_batch = min(batch_size, total_states - start)
        x = _pm1_batch_from_ints(start, cur_batch, int(n_qubits), device=device, dtype=input_dtype)
        chunks.append(model(x))
    psi = torch.cat(chunks, dim=0)
    norm = torch.linalg.norm(psi)
    if float(norm) == 0.0:
        raise ValueError("Model produced a zero statevector.")
    return psi / norm


# ---------------------------------------------------------------------------
# Constructive states
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_fixed_hamming_weight_state(
    n_qubits: int,
    hamming_weight: int,
    device: torch.device | str | None = None,
    batch_exp: int = 18,
) -> torch.Tensor:
    """Build the normalized state supported on basis strings of fixed Hamming weight."""
    if not (0 <= hamming_weight <= n_qubits):
        raise ValueError("hamming_weight must be between 0 and n_qubits.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    total_states = 1 << n_qubits
    batch_size = 1 << min(n_qubits, batch_exp)
    pieces: list[torch.Tensor] = []

    for start in range(0, total_states, batch_size):
        cur = min(batch_size, total_states - start)
        ints = torch.arange(start, start + cur, device=device, dtype=torch.int64)
        popcount = torch.zeros_like(ints)
        for bit in range(n_qubits):
            popcount += (ints >> bit) & 1
        amp = (popcount == hamming_weight).to(torch.complex128)
        pieces.append(amp)

    psi = torch.cat(pieces, dim=0)
    psi = psi / torch.linalg.norm(psi)
    return psi


def build_half_hamming_state(
    n_qubits: int,
    device: torch.device | str | None = None,
    batch_exp: int = 18,
) -> torch.Tensor:
    """Convenience wrapper for the half-Hamming state (N must be even)."""
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even for half-Hamming construction.")
    return build_fixed_hamming_weight_state(
        n_qubits=n_qubits, hamming_weight=n_qubits // 2, device=device, batch_exp=batch_exp,
    )


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def page_entropy_nats(n_a: int, n_b: int) -> float:
    """Page entropy for a 2^n_a by 2^n_b bipartition, in nats."""
    if n_a < 0 or n_b < 0:
        raise ValueError("Subsystem sizes must be non-negative.")
    if n_a > n_b:
        n_a, n_b = n_b, n_a
    m = 2**n_a
    n = 2**n_b
    dg = torch.digamma
    entropy = dg(torch.tensor(m * n + 1.0, dtype=torch.float64))
    entropy -= dg(torch.tensor(n + 1.0, dtype=torch.float64))
    entropy -= (m - 1) / (2.0 * n)
    return float(entropy)


def dicke_entropy_nats(n_qubits: int, subsystem_size: int, excitations: int | None = None) -> float:
    """Exact entropy for a Dicke state reduced to a subsystem."""
    if n_qubits % 2 != 0 and excitations is None:
        raise ValueError("Default excitations=N//2 is only meaningful for even N.")
    if excitations is None:
        excitations = n_qubits // 2
    if not (0 <= subsystem_size <= n_qubits):
        raise ValueError("subsystem_size must be in [0, n_qubits].")
    if not (0 <= excitations <= n_qubits):
        raise ValueError("excitations must be in [0, n_qubits].")

    denom = math.comb(n_qubits, excitations)
    probs: list[float] = []
    for local_exc in range(subsystem_size + 1):
        remaining_exc = excitations - local_exc
        if 0 <= remaining_exc <= (n_qubits - subsystem_size):
            p = (
                math.comb(subsystem_size, local_exc)
                * math.comb(n_qubits - subsystem_size, remaining_exc)
                / denom
            )
            probs.append(p)
    arr = np.array(probs, dtype=float)
    arr /= arr.sum()
    nz = arr > 0.0
    return float(-(arr[nz] * np.log(arr[nz])).sum())


def subsystem_entropy_from_tensor(psi_tensor: torch.Tensor, subsystem: Sequence[int]) -> float:
    """Entropy S_A from an N-index tensor psi_tensor with shape (2,)*N."""
    n_qubits = psi_tensor.dim()
    subsystem_list = list(subsystem)
    complement = [i for i in range(n_qubits) if i not in subsystem_list]
    schmidt_matrix = psi_tensor.permute(*(subsystem_list + complement)).reshape(
        2 ** len(subsystem_list), -1
    )
    singular_values = torch.linalg.svdvals(schmidt_matrix)
    lambdas = singular_values * singular_values
    lambdas = lambdas / lambdas.sum()
    lambdas = lambdas.clamp_min(1e-15)
    return float(-(lambdas * torch.log(lambdas)).sum())


def subsystem_entropy_from_state(
    psi: torch.Tensor,
    subsystem: Sequence[int],
    n_qubits: int | None = None,
) -> float:
    """Entropy S_A from a flattened statevector."""
    if n_qubits is None:
        n_qubits = int(round(math.log2(int(psi.numel()))))
    psi_tensor = psi.reshape(*([2] * n_qubits))
    return subsystem_entropy_from_tensor(psi_tensor, subsystem)


def random_contiguous_subsystems(
    n_qubits: int,
    subsystem_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Sample contiguous cyclic windows."""
    starts = rng.integers(low=0, high=n_qubits, size=n_samples)
    windows = []
    for start in starts:
        window = ((start + np.arange(subsystem_size)) % n_qubits).tolist()
        windows.append(window)
    return windows


def random_bipartitions(
    n_qubits: int,
    subsystem_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Sample random subsets uniformly without replacement."""
    subsets = []
    for _ in range(n_samples):
        subset = sorted(rng.choice(n_qubits, size=subsystem_size, replace=False).tolist())
        subsets.append(subset)
    return subsets


def sampled_entropy_curve(
    psi: torch.Tensor,
    n_qubits: int,
    subsystem_sizes: Iterable[int],
    sampler: Callable[[int], list[list[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std entropy at each subsystem size using sampled subsets."""
    psi_tensor = psi.reshape(*([2] * n_qubits))
    means: list[float] = []
    stds: list[float] = []
    for size in subsystem_sizes:
        subsets = sampler(size)
        entropies = np.array(
            [subsystem_entropy_from_tensor(psi_tensor, subset) for subset in subsets],
            dtype=float,
        )
        means.append(float(entropies.mean()))
        stds.append(float(entropies.std(ddof=1)) if len(entropies) > 1 else 0.0)
    return np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

@dataclass
class CurveStats:
    """Entropy curve mean/std and raw per-trial samples."""

    sizes: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    trials: np.ndarray


def _pick_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _summarize_trials(sizes: np.ndarray, trial_curves: list[np.ndarray]) -> CurveStats:
    trial_arr = np.stack(trial_curves, axis=0)
    return CurveStats(
        sizes=sizes.astype(int),
        mean=trial_arr.mean(axis=0),
        std=trial_arr.std(axis=0, ddof=1) if trial_arr.shape[0] > 1 else np.zeros_like(trial_arr[0]),
        trials=trial_arr,
    )


def run_single_nonlinearity_curve_trials(
    *,
    n_qubits: int,
    activations: list[str],
    mode: str = "real",
    n_trials: int = 20,
    n_translations: int = 5,
    subsystem_sizes: np.ndarray | None = None,
    use_fan_in: bool = False,
    weight_std: float = 1.0,
    bias_std: float = 1.0,
    seed: int = 1234,
    device: str | torch.device | None = None,
    batch_exp: int = 14,
) -> dict[str, CurveStats]:
    """Run single-global-nonlinearity curves for a list of activations."""
    dev = _pick_device(device)
    if subsystem_sizes is None:
        subsystem_sizes = np.arange(1, n_qubits // 2 - 1, dtype=int)

    out: dict[str, CurveStats] = {}
    for act_idx, activation in enumerate(activations):
        curves: list[np.ndarray] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + 10_000 * act_idx + trial)
            cfg = SingleNonlinearityConfig(
                n_qubits=n_qubits, activation=activation, mode=mode,
                use_fan_in=use_fan_in, weight_std=weight_std, bias_std=bias_std,
            )
            model = SingleNonlinearityNQS(cfg).to(dev)
            psi = build_normalized_state(model, n_qubits=n_qubits, device=dev, batch_exp=batch_exp)
            rng = np.random.default_rng(seed + 1000 * trial + 73 * act_idx)
            means, _ = sampled_entropy_curve(
                psi, n_qubits=n_qubits, subsystem_sizes=subsystem_sizes,
                sampler=lambda size: random_contiguous_subsystems(
                    n_qubits=n_qubits, subsystem_size=int(size), n_samples=n_translations, rng=rng,
                ),
            )
            curves.append(means)
        out[activation] = _summarize_trials(subsystem_sizes, curves)
    return out


def run_mlp_curve_trials(
    *,
    n_qubits: int,
    width: int,
    depth: int,
    activations: list[str],
    n_trials: int = 20,
    n_translations: int = 5,
    subsystem_sizes: np.ndarray | None = None,
    weight_std: float = 1.0,
    bias_std: float = 0.1,
    phase_magnitude: bool = False,
    alpha_amp: float = 0.0,
    seed: int = 1234,
    device: str | torch.device | None = None,
    batch_exp: int = 14,
) -> dict[str, CurveStats]:
    """Run MLP entropy curves for each activation."""
    dev = _pick_device(device)
    if subsystem_sizes is None:
        subsystem_sizes = np.arange(1, n_qubits // 2 - 1, dtype=int)

    out: dict[str, CurveStats] = {}
    for act_idx, activation in enumerate(activations):
        curves: list[np.ndarray] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + 10_000 * act_idx + trial)
            model = MLPNQS(
                n_qubits=n_qubits, width=width, depth=depth, activation=activation,
                weight_std=weight_std, bias_std=bias_std,
                phase_magnitude=phase_magnitude, alpha_amp=alpha_amp,
            ).to(dev)
            psi = build_normalized_state(model, n_qubits=n_qubits, device=dev, batch_exp=batch_exp)
            rng = np.random.default_rng(seed + 1000 * trial + 73 * act_idx)
            means, _ = sampled_entropy_curve(
                psi, n_qubits=n_qubits, subsystem_sizes=subsystem_sizes,
                sampler=lambda size: random_contiguous_subsystems(
                    n_qubits=n_qubits, subsystem_size=int(size), n_samples=n_translations, rng=rng,
                ),
            )
            curves.append(means)
        out[activation] = _summarize_trials(subsystem_sizes, curves)
    return out


def run_transformer_curve_trials(
    *,
    n_qubits: int,
    patch_size: int,
    stride: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff_ratio: float,
    activations: list[str],
    alpha_amp: float = 0.0,
    freeze_random_heads: bool = False,
    n_trials: int = 20,
    n_translations: int = 5,
    subsystem_sizes: np.ndarray | None = None,
    weight_std: float = 1.0,
    bias_std: float = 0.2,
    seed: int = 2026,
    device: str | torch.device | None = None,
    batch_exp: int = 14,
) -> dict[str, CurveStats]:
    """Run patch-transformer entropy curves for each activation."""
    dev = _pick_device(device)
    if subsystem_sizes is None:
        subsystem_sizes = np.arange(1, n_qubits // 2 - 1, dtype=int)

    out: dict[str, CurveStats] = {}
    for act_idx, activation in enumerate(activations):
        curves: list[np.ndarray] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + 10_000 * act_idx + trial)
            model = PatchTransformerNQS(
                n_qubits=n_qubits, patch_size=patch_size, stride=stride,
                d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff_ratio=d_ff_ratio,
                activation=activation, alpha_amp=alpha_amp,
                weight_std=weight_std, bias_std=bias_std,
                freeze_random_heads=freeze_random_heads,
            ).to(dev)
            psi = build_normalized_state(model, n_qubits=n_qubits, device=dev, batch_exp=batch_exp)
            rng = np.random.default_rng(seed + 1000 * trial + 73 * act_idx)
            means, _ = sampled_entropy_curve(
                psi, n_qubits=n_qubits, subsystem_sizes=subsystem_sizes,
                sampler=lambda size: random_contiguous_subsystems(
                    n_qubits=n_qubits, subsystem_size=int(size), n_samples=n_translations, rng=rng,
                ),
            )
            curves.append(means)
        out[activation] = _summarize_trials(subsystem_sizes, curves)
    return out


def run_random_feature_curve_trials(
    *,
    n_qubits: int,
    hidden_sizes: list[int],
    activation: str,
    n_trials: int = 20,
    n_translations: int = 5,
    subsystem_sizes: np.ndarray | None = None,
    sigma_a: float = 1.0,
    sigma_w: float = 10.0,
    bias_distribution: str = "uniform_pi",
    seed: int = 1234,
    device: str | torch.device | None = None,
    batch_exp: int = 14,
) -> dict[int, CurveStats]:
    """Run CosNet/TanhNet style curves for each hidden size."""
    dev = _pick_device(device)
    if subsystem_sizes is None:
        subsystem_sizes = np.arange(2, n_qubits // 2, dtype=int)

    out: dict[int, CurveStats] = {}
    for h_idx, hidden in enumerate(hidden_sizes):
        curves: list[np.ndarray] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + 10_000 * h_idx + trial)
            model = RandomFeatureNQS(
                n_qubits=n_qubits, hidden_units=hidden, activation=activation,
                sigma_a=sigma_a, sigma_w=sigma_w, bias_distribution=bias_distribution,
            ).to(dev)
            psi = build_normalized_state(model, n_qubits=n_qubits, device=dev, batch_exp=batch_exp)
            rng = np.random.default_rng(seed + 1000 * trial + 73 * h_idx)
            means, _ = sampled_entropy_curve(
                psi, n_qubits=n_qubits, subsystem_sizes=subsystem_sizes,
                sampler=lambda size: random_contiguous_subsystems(
                    n_qubits=n_qubits, subsystem_size=int(size), n_samples=n_translations, rng=rng,
                ),
            )
            curves.append(means)
        out[int(hidden)] = _summarize_trials(subsystem_sizes, curves)
    return out


def run_random_feature_hidden_sweep(
    *,
    n_qubits: int,
    hidden_sizes: list[int],
    activation: str,
    subsystem_size: int | None = None,
    n_trials: int = 20,
    n_translations: int = 5,
    sigma_a: float = 1.0,
    sigma_w: float = 10.0,
    bias_distribution: str = "uniform_pi",
    seed: int = 1234,
    device: str | torch.device | None = None,
    batch_exp: int = 14,
) -> dict[str, Any]:
    """Run SA(|A|=m_fixed) vs hidden-unit count for random-feature models."""
    dev = _pick_device(device)
    if subsystem_size is None:
        subsystem_size = int(round(n_qubits / 3))

    mean_vals: list[float] = []
    std_vals: list[float] = []
    for h_idx, hidden in enumerate(hidden_sizes):
        trial_vals: list[float] = []
        for trial in range(n_trials):
            torch.manual_seed(seed + 10_000 * h_idx + trial)
            model = RandomFeatureNQS(
                n_qubits=n_qubits, hidden_units=hidden, activation=activation,
                sigma_a=sigma_a, sigma_w=sigma_w, bias_distribution=bias_distribution,
            ).to(dev)
            psi = build_normalized_state(model, n_qubits=n_qubits, device=dev, batch_exp=batch_exp)
            psi_tensor = psi.reshape(*([2] * n_qubits))
            rng = np.random.default_rng(seed + 1000 * trial + 73 * h_idx)
            subsets = random_contiguous_subsystems(
                n_qubits=n_qubits, subsystem_size=subsystem_size, n_samples=n_translations, rng=rng,
            )
            entropies = [subsystem_entropy_from_tensor(psi_tensor, subset) for subset in subsets]
            trial_vals.append(float(np.mean(entropies)))
        arr = np.array(trial_vals, dtype=float)
        mean_vals.append(float(arr.mean()))
        std_vals.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

    return {
        "hidden_sizes": np.array(hidden_sizes, dtype=int),
        "subsystem_size": int(subsystem_size),
        "mean": np.array(mean_vals, dtype=float),
        "std": np.array(std_vals, dtype=float),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    return value


def save_curve_dict(
    out_prefix: str | Path,
    curves: dict[str, CurveStats] | dict[int, CurveStats],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save curve outputs to <prefix>.npz and <prefix>.json."""
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    json_curves: dict[str, Any] = {}
    for key, stats in curves.items():
        tag = str(key)
        arrays[f"{tag}_sizes"] = stats.sizes
        arrays[f"{tag}_mean"] = stats.mean
        arrays[f"{tag}_std"] = stats.std
        arrays[f"{tag}_trials"] = stats.trials
        json_curves[tag] = {
            "sizes": stats.sizes, "mean": stats.mean, "std": stats.std,
            "n_trials": int(stats.trials.shape[0]),
        }

    npz_path = prefix.with_suffix(".npz")
    np.savez(npz_path, **arrays)

    json_path = prefix.with_suffix(".json")
    payload = {"metadata": metadata, "curves": json_curves}
    json_path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))
    return npz_path, json_path


def save_npz_and_json(
    out_prefix: str | Path,
    arrays: dict[str, np.ndarray | int | float],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save arbitrary arrays plus metadata."""
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = prefix.with_suffix(".npz")
    np.savez(npz_path, **arrays)

    json_path = prefix.with_suffix(".json")
    payload = {"metadata": metadata, "arrays": {k: _to_jsonable(v) for k, v in arrays.items()}}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return npz_path, json_path


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def apply_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 14, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 140, "savefig.dpi": 300,
        "axes.spines.right": False, "axes.spines.top": False,
    })


def setup_log_x_integer_ticks(
    ax: plt.Axes,
    m_vals: Iterable[int],
    explicit_ticks: list[int] | None = None,
) -> None:
    """Log x-scale with plain integer tick labels."""
    m_arr = np.array(list(m_vals), dtype=int)
    if m_arr.size == 0:
        raise ValueError("m_vals cannot be empty.")
    ax.set_xscale("log")
    max_m = int(m_arr.max())
    if explicit_ticks is None:
        candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, max_m]
        explicit_ticks = [t for t in candidates if t <= max_m]
    ax.set_xticks(explicit_ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    for xi in range(1, max_m + 1):
        ax.axvline(xi, color="0.9", lw=0.6, alpha=0.5, zorder=0)
