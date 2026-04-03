# MorphoNAS-PL

**Plasticity in Embryogenic Neural Architecture Search**

This repository contains the code and data for our experiments on Hebbian
plasticity in recurrent networks grown by
[MorphoNAS](https://github.com/sergemedvid/MorphoNAS).

The public release is organized around nine experiments:

1. **Plasticity Characterisation** (CartPole) — Does plasticity help, and does it depend on baseline network quality? Is there a universal optimal learning rate, or does each network need its own?

2. **Within-Lifetime Adaptation** (CartPole) — Can plasticity handle mid-episode physics changes that fixed weights cannot?

3. **Adaptation Control** (CartPole) — Is the benefit genuine within-lifetime adaptation or merely robustification? Plasticity is disabled during Phase 1 and enabled only at the physics switch.

4. **Dose-Response** (CartPole) — Does giving plasticity more time after the switch improve Phase 2 performance? Tests OFF→ON at multiple switch times.

5. **Acrobot Replication** — Do the core findings generalise to a harder control task? Includes static sweep, non-stationary sweep, extended validation (248-point grid), and temporal traces.

6. **Co-Evolution** (CartPole) — Does co-evolving plasticity parameters (η, λ) alongside the developmental genome outperform fixed plasticity or no plasticity? Three conditions: (A) standard MorphoNAS GA with frozen weights, (B) GA with fixed anti-Hebbian plasticity during fitness evaluation, (C) GA with plasticity parameters encoded in the genome and evolved alongside architecture. 30 independent runs per condition.

7. **Random RNN Control** (CartPole) — Is the topology-dependence of plasticity specific to morphogenetically grown networks, or does it arise in any RNN? Generates random directed graphs matching the topology stats (neuron count, edge count) of competent MorphoNAS networks but with random (non-developmental) weights, then applies the same plasticity sweep.

8. **Co-Evolution — Acrobot** — Same three-condition co-evolution design as Experiment 6, replicated on Acrobot-v1 (6D observations, 3 actions, 20×20 developmental grid). Tests whether the co-evolution findings generalise to a harder control task with different plasticity parameter ranges (η ∈ [−0.1, +0.1], fixed η = −0.001, λ = 0.05).

9. **Analysis** — Convergence curves, evolved η distributions, Mann-Whitney comparisons, structural analysis (B1/B1_acrobot), and developmental-vs-random topology comparison (B2).

Earlier predecessor benchmarks (`B0.1`–`B0.4`) informed the final rule design
but are intentionally excluded from this public release. This repository keeps
only the final reproducible path: `B0.5`/`B0.5+` (Experiment 1), `B0.6`
(Experiment 2), `B0.6_adaptation`/`B0.6_dose_response` (Experiments 3–4),
`acrobot` (Experiment 5), `B1` (Experiment 6), `B2` (Experiment 7),
and `B1_acrobot` (Experiment 8).

---

## Installation

Requires Python `>=3.11,<3.14` as defined in `pyproject.toml`.
The public release was tested with Python `3.12.12` and `3.13.12`.

```bash
git clone https://github.com/ukma-morphonas-lab/MorphoNAS-PL.git
cd MorphoNAS-PL
python -m venv .venv
.venv/bin/pip install -e .
```

Then run scripts with `.venv/bin/python`.

---

## Repository layout

- `code/MorphoNAS/` — core developmental NAS engine

- `code/MorphoNAS_PL/` — plasticity layer, experiment modules, and analysis utilities

- `experiments/B0.5/` — Plasticity Characterisation: primary sweep (50K networks × 75 grid = 3.75M evaluations)

- `experiments/B0.5+/` — Plasticity Characterisation: extended validation (2,862 networks × 248 grid = 710K evaluations), plus genome prediction, topology regression, anti-Hebbian mechanism, and ceiling-corrected analyses

- `experiments/B0.6/` — Within-Lifetime Adaptation: non-stationary CartPole (2,362 networks, 10× pole mass and 2× gravity)

- `experiments/B0.6_adaptation/` — Adaptation Control: plasticity OFF→ON experiment (same networks, both variants)

- `experiments/B0.6_dose_response/` — Dose-Response: variable switch times (100, 200, 300, 400) with OFF→ON plasticity

- `experiments/acrobot/` — Acrobot Replication: 5,000-network pool, static and non-stationary sweeps (2× link mass), extended validation (248-point dense grid), temporal traces (per-network oracle, static, and no-plasticity conditions)

- `experiments/B1/` — Co-Evolution (CartPole): 30 GA runs × 3 conditions (A: no plasticity, B: fixed plasticity, C: co-evolved η+λ). Per-run: convergence JSONL, checkpoints, final best/population

- `experiments/B1_acrobot/` — Co-Evolution (Acrobot): same design as B1, replicated on Acrobot-v1 with 20×20 grid and task-appropriate plasticity ranges

- `experiments/B2/` — Random RNN Control: pool of random directed graphs matching MorphoNAS topology stats, plus 75-point plasticity sweep on competent random RNNs

- `scripts/` — reproduction and analysis entry points

---

## Dependency chain

When re-running experiments from scratch (rather than using committed data),
some scripts depend on outputs from earlier stages. The diagram below shows
the required order:

```
B0.5 pool generation          (run_B0_5_natural_pool.py)
 ├─► B0.5 coarse sweep        (run_B0_5_grid_sweep.py)  ← produces JSON
 │    └─► migrate to parquet  (migrate_sweep_to_parquet.py)
 ├─► B0.5+ subsample setup    (setup_B0_5plus.py)  ← needs B0.5 sweep parquets
 │    └─► B0.5+ fine sweep    (run_B0_5_grid_sweep.py + consolidate_B0_5plus.py)
 │    └─► B0.6 NS sweep       (run_B0_6_nonstationary.py)
 │    └─► B0.6 temporal       (profile_B0_6_nonstationary.py)
 │    └─► B0.6 adaptation     (run_B0_6_adaptation.py)
 │    └─► B0.6 dose-response  (run_B0_6_dose_response.py)
 └─► B2 random RNN pool       (run_B2_random_rnn_pool.py)
      └─► B2 sweep            (run_B2_random_rnn_sweep.py)

Acrobot pool generation        (run_acrobot_pool.py)
 ├─► Acrobot static sweep      (run_acrobot_sweep.py)
 ├─► Acrobot non-stationary    (run_acrobot_nonstationary.py)
 ├─► Acrobot extended          (run_acrobot_extended.py)
 └─► Acrobot temporal oracle   (run_acrobot_temporal_oracle.py)  ← must run first
      ├─► Acrobot temporal NS no-plasticity  (run_acrobot_temporal_ns_noplasticity.py)
      └─► Acrobot temporal static            (run_acrobot_temporal_static.py)

B1 co-evolution                (run_B1_coevolution.py)  — standalone, no dependencies
```

Scripts at the top of each chain can run from scratch; downstream scripts
require their upstream outputs to exist. The B0.6 and Acrobot non-stationary /
extended / temporal scripts hardcode their pool paths and have no `--pool-dir`
CLI override — they expect the committed data at the default locations.

---

## Reproducing the experiments

### Quick start — regenerate figures only

All sweep results are already committed. No experiments need to be re-run.

```bash
# Plasticity Characterisation (B0.5+) — headline impact, heatmaps, regret, gate finding
.venv/bin/python scripts/analyze_B0_5plus_comprehensive.py \
    --sweep-dir  experiments/B0.5+/sweep \
    --pool-dir   experiments/B0.5+/pool_subsample \
    --output-dir experiments/B0.5+/analysis

.venv/bin/python scripts/analyze_B0_5plus_gate_simulation.py \
    --sweep-dir  experiments/B0.5+/sweep \
    --output-dir experiments/B0.5+/analysis

# Within-Lifetime Adaptation (B0.6) — adaptation premium, cross-variant comparison
.venv/bin/python scripts/analyze_B0_6_cross_variant.py

# Acrobot Replication — static and non-stationary analysis
.venv/bin/python scripts/analyze_acrobot_static_sweep.py
.venv/bin/python scripts/analyze_acrobot_ns_sweep.py

# Acrobot Extended Validation — dense grid analysis
.venv/bin/python scripts/analyze_acrobot_extended.py --phase dense
```

### Paper figures

Generate all camera-ready figures (requires analysis outputs from the Quick
Start commands above):

```bash
.venv/bin/python scripts/generate_paper_figures.py
```

Generate the MorphoNAS developmental time-series figure (Figure 1):

```bash
.venv/bin/python scripts/generate_development_figure.py
```

### Additional analyses (use existing data, no experiments needed)

```bash
# Genome-based prediction of plasticity benefit
.venv/bin/python scripts/analyze_genome_prediction.py

# Deeper topology regression (density reversal quantification)
.venv/bin/python scripts/analyze_topology_regression.py

# Anti-Hebbian mechanism hypothesis testing
.venv/bin/python scripts/analyze_anti_hebbian_mechanism.py

# Ceiling-corrected effect sizes (fair cross-stratum comparison)
.venv/bin/python scripts/analyze_ceiling_corrected.py
```

### Plasticity Characterisation, Stage 1 (B0.5) — re-run the coarse sweep from scratch

The full pool of 50 000 network genomes is stored in a single parquet file
(`experiments/B0.5/pool_natural/pool_natural.parquet`, 22 MB) rather than
50 000 individual JSON files. To extract them before re-running:

```bash
.venv/bin/python scripts/extract_pool.py   # writes experiments/B0.5/pool_natural/networks/
```

To regenerate the pool itself from scratch (takes significant compute time):

```bash
.venv/bin/python scripts/run_B0_5_natural_pool.py \
    --output-dir experiments/B0.5/pool_natural \
    --max-seeds  50000 \
    --start-seed 42
```

The committed B0.5 sweep is stored as four parquet parts in
`experiments/B0.5/sweep/` so each file stays under GitHub's 100 MB limit.
To re-run the coarse 75-point sweep from scratch (50K networks, ~100h on 10 cores):

```bash
# Step 1: Run the sweep (produces per-network JSON files)
.venv/bin/python scripts/run_B0_5_grid_sweep.py \
    --pool-dir   experiments/B0.5/pool_natural \
    --output-dir experiments/B0.5/sweep

# Step 2: Convert JSON results to parquet (required by downstream scripts)
.venv/bin/python scripts/migrate_sweep_to_parquet.py
# Then: mv experiments/B0.5/sweep_parquet experiments/B0.5/sweep
```

### Plasticity Characterisation, Stage 2 (B0.5+) — re-run the fine η×λ grid sweep

The 2,862 subsample networks are already in `experiments/B0.5+/pool_subsample/networks/`
and the committed sweep is stored as one merged parquet in `experiments/B0.5+/sweep/`.

To regenerate the subsample from a B0.5 pool (requires `experiments/B0.5/` to exist):

```bash
.venv/bin/python scripts/setup_B0_5plus.py
```

This selects all non-Weak networks plus a random sample of 500 Weak networks,
copies them to `experiments/B0.5+/pool_subsample/networks/`, and extracts the
matching rows from the B0.5 sweep into `experiments/B0.5+/sweep/`.

To re-run the fine 248-point sweep:

```bash
# Run the extended grid (produces JSON)
.venv/bin/python scripts/run_B0_5_grid_sweep.py \
    --pool-dir   experiments/B0.5+/pool_subsample \
    --output-dir experiments/B0.5+/sweep_extended

# Consolidate extended JSON results into sweep/ as parquet
.venv/bin/python scripts/consolidate_B0_5plus.py
```

### Within-Lifetime Adaptation (B0.6) — non-stationary sweep and analysis

**Prerequisite:** requires `experiments/B0.5+/pool_subsample/networks/` (see B0.5+ section above).

To re-run the B0.6 non-stationary sweep from scratch (~4h on 10 cores):

```bash
# Run both variants
.venv/bin/python scripts/run_B0_6_nonstationary.py --variant all

# Or one at a time
.venv/bin/python scripts/run_B0_6_nonstationary.py --variant gravity_2x
.venv/bin/python scripts/run_B0_6_nonstationary.py --variant heavy_pole

# Pilot (10 networks per stratum)
.venv/bin/python scripts/run_B0_6_nonstationary.py --pilot
```

Pre-computed sweep results are in `experiments/B0.6/sweep/`. To regenerate the
per-variant analyses:

```bash
.venv/bin/python scripts/analyze_B0_6_nonstationary.py \
    --variant    gravity_2x \
    --sweep-dir  experiments/B0.6/sweep \
    --output-dir experiments/B0.6/analysis_gravity_2x

.venv/bin/python scripts/analyze_B0_6_nonstationary.py \
    --variant    heavy_pole \
    --sweep-dir  experiments/B0.6/sweep \
    --output-dir experiments/B0.6/analysis

.venv/bin/python scripts/analyze_B0_6_cross_variant.py \
    --output-dir experiments/B0.6/analysis_cross_variant
```

The public repo also includes the derived per-variant summaries
(`experiments/B0.6/analysis/`, `experiments/B0.6/analysis_gravity_2x/`)
and temporal traces (`experiments/B0.6/temporal_profile/`) so that
`scripts/analyze_B0_6_cross_variant.py` works out of the box.

To regenerate the temporal traces from scratch (~2h on 10 cores):

```bash
.venv/bin/python scripts/profile_B0_6_nonstationary.py

# Quick test (10 networks per stratum, 5 rollouts)
.venv/bin/python scripts/profile_B0_6_nonstationary.py --per-stratum 10 --rollouts 5
```

### Adaptation Control (B0.6_adaptation) — plasticity OFF→ON

**Prerequisite:** requires `experiments/B0.5+/pool_subsample/networks/` (see B0.5+ section above).

Disables plasticity during Phase 1 (steps 0–199), enables at the physics
switch point. Tests whether the benefit is genuine adaptation or
robustification.

```bash
# Full run, both variants (~3h on 10 cores)
.venv/bin/python scripts/run_B0_6_adaptation.py --variant all

# Pilot (10 networks per stratum)
.venv/bin/python scripts/run_B0_6_adaptation.py --pilot --variant heavy_pole
```

### Dose-Response (B0.6_dose_response) — variable switch times

**Prerequisite:** requires `experiments/B0.5+/pool_subsample/networks/` (see B0.5+ section above).

Tests OFF→ON plasticity at switch steps 100, 200, 300, 400 on the
anti-Hebbian range (η < 0, λ = 0.01).

```bash
# Full run (~2h on 10 cores)
.venv/bin/python scripts/run_B0_6_dose_response.py

# Custom switch times
.venv/bin/python scripts/run_B0_6_dose_response.py --switch-times "100,200,300,400"
```

### Acrobot Replication — re-run from scratch

5,000 random genomes on a 20×20 developmental grid, evaluated on Acrobot-v1.
Pre-computed sweep results are in `experiments/acrobot/`. The pool network
genomes are in `experiments/acrobot/pool/networks/` (5,000 JSON files).

The pool must be generated first — the non-stationary sweep, extended
validation, and temporal trace scripts all require
`experiments/acrobot/pool/networks/` to exist (hardcoded, no CLI override).

```bash
# Regenerate pool (takes ~30 min)
.venv/bin/python scripts/run_acrobot_pool.py \
    --target-valid 5000 \
    --output-dir   experiments/acrobot/pool

# Static plasticity sweep (22-point grid on non-Weak networks, ~2.5h)
.venv/bin/python scripts/run_acrobot_sweep.py \
    --pool-dir   experiments/acrobot/pool \
    --output-dir experiments/acrobot/sweep_static

# Non-stationary sweep (2× lower-link mass at step 50, ~4.5h)
# Prerequisite: acrobot pool above
.venv/bin/python scripts/run_acrobot_nonstationary.py \
    --variant heavy_link2_2x \
    --switch-step 50

# Analysis
.venv/bin/python scripts/analyze_acrobot_static_sweep.py
.venv/bin/python scripts/analyze_acrobot_ns_sweep.py
```

### Acrobot Extended Validation — fine η resolution

**Prerequisite:** requires `experiments/acrobot/pool/networks/` (see Acrobot Replication above).

Two-phase experiment testing whether finer η resolution reveals beneficial
fixed settings that the coarse 22-point grid missed.

```bash
# Phase 1: Monte Carlo pilot (30 sampled grid points, ~1h)
.venv/bin/python scripts/run_acrobot_extended.py --phase pilot

# Phase 2: Dense 248-point grid (~20h on 10 cores)
.venv/bin/python scripts/run_acrobot_extended.py --phase dense \
    --eta-min -0.1 --eta-max 0.1 --n-eta 31 \
    --lambda-min 0 --lambda-max 0.1 --n-lambda 8

# Analysis
.venv/bin/python scripts/analyze_acrobot_extended.py --phase dense
```

### Acrobot Temporal Traces

**Prerequisite:** requires `experiments/acrobot/pool/networks/` and the
non-stationary sweep parquet (see Acrobot Replication above).

Per-timestep |Δw|, observation, and action traces for 200 networks
(50 per stratum) under three conditions: non-stationary with per-network
oracle plasticity, static with oracle plasticity, and non-stationary
without plasticity.

The oracle script must run first — it produces `metadata_acrobot.json`
which the other two scripts require.

```bash
# Step 1 (must run first): NS + per-network oracle plasticity
.venv/bin/python scripts/run_acrobot_temporal_oracle.py

# Step 2 (depends on step 1): NS + no plasticity (baseline)
.venv/bin/python scripts/run_acrobot_temporal_ns_noplasticity.py

# Step 2 (depends on step 1): Static traces only
.venv/bin/python scripts/run_acrobot_temporal_static.py
```

### Co-Evolution (B1) — evolve architecture with/without plasticity

Three conditions compared over 30 independent GA runs each (population 50,
200 generations, 10×10 grid, 3 morphogens):

- **Condition A**: Standard MorphoNAS evolution, frozen weights
- **Condition B**: MorphoNAS evolution, fitness evaluated with fixed anti-Hebbian plasticity (η = −0.01, λ = 0.01)
- **Condition C**: Extended genome co-evolves plasticity parameters (η ∈ [−0.5, +0.5], λ ∈ [0, 0.1]) alongside architecture

```bash
# Full experiment: 30 runs per condition (~6–7h per run with 8 workers)
.venv/bin/python scripts/run_B1_coevolution.py --condition A --run-ids 0-29 --resume
.venv/bin/python scripts/run_B1_coevolution.py --condition B --run-ids 0-29 --resume
.venv/bin/python scripts/run_B1_coevolution.py --condition C --run-ids 0-29 --resume

# Partition across AWS instances (example: 3 machines per condition)
.venv/bin/python scripts/run_B1_coevolution.py --condition A --run-ids 0-9 --resume
.venv/bin/python scripts/run_B1_coevolution.py --condition A --run-ids 10-19 --resume
.venv/bin/python scripts/run_B1_coevolution.py --condition A --run-ids 20-29 --resume

# Quick smoke test
.venv/bin/python scripts/run_B1_coevolution.py --condition C --run-ids 0 \
    --pop-size 10 --max-gen 5 --num-rollouts 2

# Analysis (convergence curves, eta distribution, Mann-Whitney, structural comparison)
.venv/bin/python scripts/analyze_B1_coevolution.py \
    --input-dir  experiments/B1 \
    --output-dir experiments/B1/analysis
```

Each run produces `generations.jsonl` (per-generation stats), `checkpoint.json`
(for resume), `final_best.json`, and `final_population.json`.

### Co-Evolution — Acrobot (B1_acrobot)

Same three-condition design as CartPole B1, replicated on Acrobot-v1 with
task-appropriate parameters: 20×20 developmental grid, fixed η = −0.001 / λ = 0.05
(Condition B), evolved η ∈ [−0.1, +0.1] (Condition C).

```bash
# Full experiment: 30 runs per condition
.venv/bin/python scripts/run_B1_coevolution.py --env acrobot --condition A --run-ids 0-29 --resume
.venv/bin/python scripts/run_B1_coevolution.py --env acrobot --condition B --run-ids 0-29 --resume
.venv/bin/python scripts/run_B1_coevolution.py --env acrobot --condition C --run-ids 0-29 --resume

# Analysis
.venv/bin/python scripts/analyze_B1_coevolution.py \
    --input-dir  experiments/B1_acrobot \
    --output-dir experiments/B1_acrobot/analysis
```

### Random RNN Control (B2) — developmental vs random topology

**Prerequisite:** requires `experiments/B0.5/pool_natural/` with network JSON files
(see B0.5 section; run `extract_pool.py` if starting from the committed parquet).

Tests whether plasticity's topology-dependence is specific to morphogenetically
grown networks. Generates random directed graphs with matching (N, E) but
non-developmental random weights.

```bash
# Stage 1: Generate pool (5 random RNNs per competent MorphoNAS network)
.venv/bin/python scripts/run_B2_random_rnn_pool.py \
    --b05-pool-dir experiments/B0.5/pool_natural \
    --output-dir   experiments/B2/pool

# Partition pool generation across machines
.venv/bin/python scripts/run_B2_random_rnn_pool.py --start-index 0 --max-networks 1000 --resume
.venv/bin/python scripts/run_B2_random_rnn_pool.py --start-index 1000 --max-networks 1000 --resume

# Stage 2: Plasticity sweep (same 75-point grid as B0.5)
.venv/bin/python scripts/run_B2_random_rnn_sweep.py \
    --pool-path  experiments/B2/pool/random_rnn_pool.jsonl \
    --output-dir experiments/B2/sweep

# Analysis (competence rate, oracle improvement, regret, anti-Hebbian dominance)
.venv/bin/python scripts/analyze_B2_random_rnn.py \
    --b2-pool     experiments/B2/pool/random_rnn_pool.jsonl \
    --b2-sweep    experiments/B2/sweep/random_rnn_sweep.jsonl \
    --b05-sweep-dir experiments/B0.5/sweep \
    --output-dir  experiments/B2/analysis
```

---

## Plasticity rule

Each weight is updated after every propagation timestep:

$$\Delta w_{ij} = \eta \cdot x_i \cdot x_j - \lambda \cdot w_{ij}$$

- **η** (learning rate): sign controls Hebbian (η > 0) vs anti-Hebbian (η < 0); primary sweep [−0.05, +0.05] (15 levels), extended validation [−0.5, +0.5] (31 levels)

- **λ** (decay): prevents runaway weight growth; primary sweep [0, 0.01] (5 levels), extended [0, 0.1] (8 levels)

Anti-Hebbian (η < 0) significantly outperforms Hebbian for competent CartPole
networks (Cohen's d = 0.54–0.66, p < 0.001). The effect is more nuanced on
Acrobot, where the per-network optimal sign splits roughly evenly. Decay
λ = 0.01 is essential for aggressive |η|: without it, harm rates exceed 80%
at |η| > 0.2.

---

## Key results

| Question | Finding |
|----------|---------|
| Does plasticity help? | Yes, tier-dependently. Up to 93% of competent CartPole networks improve under oracle tuning, with mean gains of +60.6 reward on the primary sweep (+86.3 on the extended grid). |
| Universal rate? | No. Regret under fixed parameters reaches 52–100%. Cross-validation confirms 83–90% of oracle advantage is genuine per-network heterogeneity. |
| Real adaptation? | Yes. Under non-stationarity, 88–90% of competent CartPole networks benefit. The OFF→ON experiment confirms this is genuine adaptation, not robustification: plasticity disabled during Phase 1, enabled only at the switch point. |
| Dose-response? | Phase 2 performance varies with plasticity exposure time across switch points 100–400. |
| Cross-task generalisation? | Partially. On Acrobot, 94.3% (static) and 89.9% (NS) of non-weak networks improve under oracle tuning with fine η resolution. No fixed setting helps on average — the 248-point dense grid confirms 100% regret at the population level, a genuine difference from CartPole. |
| Genome prediction? | Genome features predict plasticity benefit with AUC = 0.63 (random forest). Optimal η sign is predictable at 80% accuracy. Cross-task transfer from CartPole to Acrobot is weak (AUC = 0.45). |
| Topology reversal? | Confirmed. Connectivity density × stratum interaction is significant (p < 0.05): denser networks benefit in Low-mid, sparser in Perfect. Interaction model R² = 0.25. |
| Ceiling correction? | After normalising for asymmetric reward ceilings, High-mid captures 86.5% of available headroom — the highest across strata. Cross-task comparison becomes meaningful on the normalised scale. |
| Co-evolution (CartPole)? | All three conditions (A, B, C) consistently evolve perfect controllers (reward = 500). Condition C co-evolves η values with mixed signs across runs — evolution does not uniformly converge to anti-Hebbian, suggesting the optimal plasticity sign is architecture-dependent even under evolutionary pressure. Full analysis via `analyze_B1_coevolution.py`. |
| Co-evolution (Acrobot)? | Acrobot is a harder benchmark: best fitness plateaus below 1.0 in 200 generations. Condition C evolves η in the anti-Hebbian range. Cross-task comparison with CartPole B1 reveals whether co-evolution benefits scale with task difficulty. |
| Developmental vs random? | Random RNNs matching MorphoNAS topology stats are 8× less likely to be competent (0.6% vs 4.7%), indicating that developmental structure provides a strong inductive bias for viable controller architectures. Plasticity sweep on the 65 competent random RNNs enables direct comparison of topology-dependence patterns. Full analysis via `analyze_B2_random_rnn.py`. |
