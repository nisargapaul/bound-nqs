# BoundOnNQS

Code for:

- Nisarga Paul, *Bound on entanglement in neural quantum states* (arXiv:2510.11797, to appear in PRL, 2026).

## Install

```bash
pip install numpy torch matplotlib
```

## Main runs

Outputs are written to `results/` as `.npz` and `.json`.
The commands below reproduce the supplemental material figures (5 translation samples per point).
The main text figures used 10 translations; increase `--translations 10` to match exactly.

```bash
# single-nonlinearity
python run_sn_curve.py --n 22 --mode real --activations tanh,sin,gelu,relu --trials 20 --translations 5 --seed 2010 --out-prefix results/sn_real
python run_sn_curve.py --n 22 --mode pure_phase --activations tanh,sin,gelu,relu --trials 20 --translations 5 --seed 2010 --out-prefix results/sn_pure_phase

# MLP
python run_mlp_curve.py --n 22 --width 3 --depth 2 --activations tanh,sin --trials 20 --translations 5 --seed 1400 --out-prefix results/mlp_w3_d2

# Transformer
python run_transformer_curve.py --n 22 --patch-size 6 --stride 5 --d-model 32 --n-heads 4 --n-layers 4 --alpha-amp 1.0 --activations tanh,sin --freeze-random-heads --trials 20 --translations 5 --seed 2036 --out-prefix results/transformer_p6_s5

# CosNet / TanhNet random-feature baselines
python run_random_feature.py --n 22 --activation cos  --hidden-sizes 1,2,5,100  --trials 20 --translations 5 --sigma-a 1.0 --sigma-w 10.0 --mode both  --out-prefix results/cosnet
python run_random_feature.py --n 22 --activation tanh --hidden-sizes 1,2,5,1000 --trials 20 --translations 5 --sigma-a 1.0 --sigma-w 10.0 --mode curve --out-prefix results/tanhnet

# analytic/constructive checks
python run_dicke_curve.py --n 22 --m-min 2 --m-max 10 --out-prefix results/dicke
python run_half_hamming_curve.py --n 22 --batch-exp 18 --out-prefix results/half_hamming
```

## Example notebook

- `example_single_nonlinearity.ipynb` shows minimal usage for single-nonlinearity entanglement scaling.
