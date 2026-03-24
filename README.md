# MorphoNAS-PL

**Plasticity in Embryogenic Neural Architecture Search**

This repository contains the code and data for our experiments on Hebbian
plasticity in recurrent networks grown by
[MorphoNAS](https://github.com/sergemedvid/MorphoNAS).

The public release is organized around three experiment blocks:

1. Does plasticity help, and does it depend on baseline network quality?
2. Is there a universal optimal learning rate, or does each network need its own?
3. Can plasticity handle mid-episode physics changes that fixed weights cannot?

Earlier predecessor benchmarks (`B0.1`–`B0.4`) informed the final rule design
but are intentionally excluded from this public release. This repository keeps
only the final reproducible path used for the main results: `B0.5`, `B0.5+`,
and `B0.6`.

---

## Installation

Requires Python `>=3.11,<3.14` as defined in `pyproject.toml`.
The public release was tested with Python `3.12.12`.
We use [uv](https://docs.astral.sh/uv/) for dependency management and command execution.

```bash
git clone https://github.com/ukma-morphonas-lab/MorphoNAS-PL.git
cd MorphoNAS-PL
uv sync
```

Then run scripts with `uv run`.

---

## Repository layout

- `code/MorphoNAS/` — core developmental NAS engine
- `code/MorphoNAS_PL/` — plasticity layer and analysis utilities
- `experiments/B0.5/` — Experiment 1, Stage 1: 50K-network coarse sweep
- `experiments/B0.5+/` — Experiment 1, Stage 2: 2,862-network fine sweep and analysis
- `experiments/B0.6/` — Experiment 2: non-stationary CartPole sweeps and temporal traces
- `scripts/` — reproduction and analysis entry points

---

## Reproducing the experiments

### Quick start — regenerate figures only

All sweep results are already committed. No experiments need to be re-run.
The B0.6 temporal traces and per-variant summaries required by the
cross-variant script are committed too.

```bash
# Experiment 1 (B0.5+, Stage 2) — headline impact, heatmaps, regret, gate finding
uv run python scripts/analyze_B0_5plus_comprehensive.py \
    --sweep-dir  experiments/B0.5+/sweep \
    --pool-dir   experiments/B0.5+/pool_subsample \
    --output-dir experiments/B0.5+/analysis

uv run python scripts/analyze_B0_5plus_gate_simulation.py \
    --sweep-dir  experiments/B0.5+/sweep \
    --output-dir experiments/B0.5+/analysis

# Experiment 2 (B0.6) — adaptation premium, cross-variant comparison
uv run python scripts/analyze_B0_6_cross_variant.py
```

### Experiment 1, Stage 1 (B0.5) — re-run the coarse sweep from scratch

The full pool of 50 000 network genomes is stored in a single parquet file
(`experiments/B0.5/pool_natural/pool_natural.parquet`, 22 MB) rather than
50 000 individual JSON files. To extract them before re-running:

```bash
uv run python scripts/extract_pool.py   # writes experiments/B0.5/pool_natural/networks/
```

To regenerate the pool itself from scratch (takes significant compute time):

```bash
uv run python scripts/run_B0_5_natural_pool.py \
    --output-dir experiments/B0.5/pool_natural \
    --max-seeds  50000 \
    --start-seed 42
```

The committed B0.5 sweep is stored as four parquet parts in
`experiments/B0.5/sweep/` so each file stays under GitHub's 100 MB limit.

### Experiment 1, Stage 2 (B0.5+) — re-run the fine η×d grid sweep

The 2 862 subsample networks are already in `experiments/B0.5+/pool_subsample/networks/`
and the committed sweep is stored as one merged parquet in `experiments/B0.5+/sweep/`.
To re-run:

```bash
uv run python scripts/run_B0_5_grid_sweep.py \
    --pool-dir   experiments/B0.5+/pool_subsample \
    --output-dir experiments/B0.5+/sweep
```

### Experiment 2 (B0.6) — regenerate non-stationary analysis

Physics changes mid-episode at step 200 (gravity ×2 or pole mass ×10).
Pre-computed sweep results are in `experiments/B0.6/sweep/`. To regenerate the
per-variant analyses:

```bash
uv run python scripts/analyze_B0_6_nonstationary.py \
    --variant    gravity_2x \
    --sweep-dir  experiments/B0.6/sweep \
    --output-dir experiments/B0.6/analysis_gravity_2x

uv run python scripts/analyze_B0_6_nonstationary.py \
    --variant    heavy_pole \
    --sweep-dir  experiments/B0.6/sweep \
    --output-dir experiments/B0.6/analysis

uv run python scripts/analyze_B0_6_cross_variant.py \
    --output-dir experiments/B0.6/analysis_cross_variant
```

The public repo also includes the derived per-variant summaries
(`experiments/B0.6/analysis/`, `experiments/B0.6/analysis_gravity_2x/`)
and temporal traces (`experiments/B0.6/temporal_profile/`) so that
`scripts/analyze_B0_6_cross_variant.py` works out of the box.

---

## Plasticity rule

Each weight is updated after every step:

$$w \leftarrow (1 - d)\,w + \eta \cdot x_{\text{post}} \cdot x_{\text{pre}}$$

- **η** (learning rate): sign controls Hebbian vs anti-Hebbian; sweep range [−0.5, +0.5]
- **d** (decay): prevents runaway growth; sweep range [0, 0.1]

Anti-Hebbian (η < 0) dominates across all network tiers. Decay d = 0.01 is
essential for aggressive |η|: without it, harm rates exceed 80% at |η| > 0.2.

---

## Key results

| Question | Finding |
|----------|---------|
| Does plasticity help? | Yes, tier-dependently. High-mid: 99% improve, mean +86 reward. |
| Universal rate? | No. Best fixed rate captures at most 40% of per-network oracle. Cross-validation confirms 74–92% of variance is real signal. |
| Real adaptation? | Yes. Gravity-2x shows amplified plasticity benefit (p < 0.001). Same networks benefit across both disruption variants (ρ = 0.65–0.74). |
