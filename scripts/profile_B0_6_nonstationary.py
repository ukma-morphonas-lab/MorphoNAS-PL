#!/usr/bin/env python3
"""
B0.6 Non-Stationary Temporal Profiling — |Δw| + Physics per Step

Runs networks under non-stationary CartPole (gravity_2x and heavy_pole)
with per-step recording of weight changes and physical observables.
Answers: does plasticity respond to the step-200 physics switch?

Records per step: |Δw|, theta, theta_dot, x_dot, action, phase.

Usage:
    .venv/bin/python scripts/profile_B0_6_nonstationary.py
    .venv/bin/python scripts/profile_B0_6_nonstationary.py --variant heavy_pole
    .venv/bin/python scripts/profile_B0_6_nonstationary.py --per-stratum 10 --rollouts 5  # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experimentB0_5_natural import (
    VERIFICATION_SEEDS,
    create_propagator,
    load_network_from_file,
    set_shutdown_event,
)
from MorphoNAS_PL.env_wrappers import MidEpisodeSwitchWrapper
from MorphoNAS_PL.experimentB0_6_nonstationary import (
    DEFAULT_SWITCH_STEP,
    VARIANTS,
)
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook
from MorphoNAS.grid import Grid

import gymnasium as gym
import networkx as nx

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
B05P_ROOT = Path("experiments/B0.5+")
B05P_POOL = B05P_ROOT / "pool_subsample" / "networks"
OUTPUT_ROOT = Path("experiments/B0.6/temporal_profile")

SWITCH_STEP = DEFAULT_SWITCH_STEP  # 200

# ── Best eta per stratum per variant (decay=0.01, from B0.6 analysis) ──
VARIANT_PARAMS = {
    "gravity_2x": {
        "low_mid":      {"eta": -0.20, "decay": 0.01},
        "high_mid":     {"eta": -0.20, "decay": 0.01},
        "near_perfect": {"eta": -0.20, "decay": 0.01},
        "perfect":      {"eta": 0.0,   "decay": 0.0},
    },
    "heavy_pole": {
        "low_mid":      {"eta": -0.20, "decay": 0.01},
        "high_mid":     {"eta": -0.10, "decay": 0.01},
        "near_perfect": {"eta": -0.01, "decay": 0.01},
        "perfect":      {"eta": 0.0,   "decay": 0.0},
    },
}

TARGET_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


def _load_networks_by_stratum(
    per_stratum: int,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Sample networks per stratum (excluding Weak)."""
    id_file = B05P_ROOT / "network_ids.txt"
    all_ids = [int(line.strip()) for line in open(id_file) if line.strip()]

    by_stratum: dict[str, list[int]] = {}
    for nid in all_ids:
        fpath = B05P_POOL / f"network_{nid:05d}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        stratum = data.get("stratum", "weak")
        if stratum in TARGET_STRATA:
            by_stratum.setdefault(stratum, []).append(nid)

    rng = np.random.default_rng(seed)
    sampled: dict[str, list[int]] = {}
    for stratum in TARGET_STRATA:
        ids = by_stratum.get(stratum, [])
        n = min(per_stratum, len(ids))
        chosen = sorted(rng.choice(ids, size=n, replace=False).tolist())
        sampled[stratum] = [int(x) for x in chosen]
        print(f"  {stratum}: {n} networks (from {len(ids)} available)")

    return sampled


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_single(args: tuple) -> dict | None:
    """Worker: run one network under one variant, recording per-step traces."""
    network_path, network_id, stratum, variant, eta, decay, rollouts = args

    try:
        genome, metadata = load_network_from_file(str(network_path))

        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        G = grid.get_graph()

        if G.number_of_nodes() < 6:
            return None

        try:
            graph_diameter = int(nx.diameter(G))
        except nx.NetworkXError:
            max_length = 0
            for source in G.nodes():
                lengths = nx.shortest_path_length(G, source)
                max_length = max(max_length, max(lengths.values()))
            graph_diameter = int(max_length)

        # Create hook (None for eta=0)
        if abs(eta) < 1e-12:
            hook = None
        else:
            hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))

        propagator = create_propagator(grid, edge_hook=hook, graph_diameter=graph_diameter)
        base_weights = propagator.get_weights().copy()

        # Create non-stationary env
        target_params = VARIANTS[variant]
        base_env = gym.make("CartPole-v1", render_mode=None)
        env = MidEpisodeSwitchWrapper(
            base_env, switch_step=SWITCH_STEP, target_params=target_params
        )

        seeds = VERIFICATION_SEEDS[:int(rollouts)]
        episode_traces: list[dict] = []
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        prop_hook = getattr(propagator, "edge_hook", None)

        for i in range(int(rollouts)):
            propagator.reset()
            propagator.set_weights(base_weights.copy())

            observation, info = env.reset(seed=int(seeds[i]))
            done = False

            # Per-step accumulators
            step_dw: list[float] = []
            step_theta: list[float] = []
            step_theta_dot: list[float] = []
            step_x_dot: list[float] = []
            step_action: list[int] = []
            step_phase: list[int] = []

            total_reward = 0.0

            while not done:
                obs = np.array(observation).flatten()
                propagator.propagate(obs)
                output_values = propagator.get_output()
                action = int(output_values.argmax().item())

                # Record pre-step observation
                step_theta.append(float(obs[2]))
                step_theta_dot.append(float(obs[3]))
                step_x_dot.append(float(obs[1]))
                step_action.append(action)

                observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                reward_f = float(reward)
                total_reward += reward_f

                if prop_hook is not None:
                    prop_hook.on_reward(reward_f)

                # Apply plasticity and record |Δw|
                if prop_hook is not None and hasattr(prop_hook, "end_step"):
                    deltas = prop_hook.end_step()
                    propagator.apply_edge_weight_deltas(deltas)
                    dw = float(deltas.abs().mean().item()) if deltas.numel() > 0 else 0.0
                else:
                    dw = 0.0
                step_dw.append(dw)

                step_phase.append(info.get("phase", 1))

            if prop_hook is not None and hasattr(prop_hook, "end_episode"):
                prop_hook.end_episode(total_reward)

            episode_traces.append({
                "dw": step_dw,
                "theta": step_theta,
                "theta_dot": step_theta_dot,
                "x_dot": step_x_dot,
                "action": step_action,
                "phase": step_phase,
            })
            episode_rewards.append(total_reward)
            episode_lengths.append(len(step_dw))

        env.close()

        return {
            "network_id": int(network_id),
            "stratum": stratum,
            "variant": variant,
            "eta": float(eta),
            "decay": float(decay),
            "traces": episode_traces,
            "rewards": episode_rewards,
            "lengths": episode_lengths,
        }
    except Exception as e:
        logger.error(f"Error evaluating network {network_id} ({variant}): {e}")
        return None


def _save_traces(all_results: dict[str, dict[str, list[dict]]], output_dir: Path):
    """Save raw traces to compressed NPZ files (one per variant)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, stratum_results in all_results.items():
        arrays: dict[str, np.ndarray] = {}
        for stratum, results in stratum_results.items():
            for r in results:
                for ep_idx, trace in enumerate(r["traces"]):
                    key_prefix = f"{stratum}_{r['network_id']}_{ep_idx}"
                    arrays[f"{key_prefix}_dw"] = np.array(trace["dw"], dtype=np.float32)
                    arrays[f"{key_prefix}_theta"] = np.array(trace["theta"], dtype=np.float32)
                    arrays[f"{key_prefix}_theta_dot"] = np.array(trace["theta_dot"], dtype=np.float32)
                    arrays[f"{key_prefix}_x_dot"] = np.array(trace["x_dot"], dtype=np.float32)
                    arrays[f"{key_prefix}_action"] = np.array(trace["action"], dtype=np.int8)
                    arrays[f"{key_prefix}_phase"] = np.array(trace["phase"], dtype=np.int8)

        path = output_dir / f"traces_{variant}.npz"
        np.savez_compressed(path, **arrays)
        n_episodes = sum(len(r["traces"]) for rs in stratum_results.values() for r in rs)
        print(f"  Saved {path.name}: {n_episodes} episodes, {len(arrays)} arrays")


def _plot_comparison(
    all_results: dict[str, dict[str, list[dict]]],
    output_dir: Path,
    bin_size: int = 10,
):
    """Generate comparative plots for gravity_2x vs heavy_pole."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    stratum_colors = {
        "low_mid": "#2196F3",
        "high_mid": "#FF9800",
        "near_perfect": "#4CAF50",
        "perfect": "#9C27B0",
    }
    stratum_labels = {
        "low_mid": "Low-mid",
        "high_mid": "High-mid",
        "near_perfect": "Near-perfect",
        "perfect": "Perfect",
    }
    variant_labels = {
        "gravity_2x": "Gravity 2x (g=19.6)",
        "heavy_pole": "Heavy Pole 10x (m=1.0)",
    }

    variants = [v for v in ["gravity_2x", "heavy_pole"] if v in all_results]
    if len(variants) < 2:
        print("  Need both variants for comparison plots. Skipping comparison.")
        return

    def _collect_binned(results_by_stratum, field, bin_size):
        """Collect and bin a per-step field across all episodes for each stratum."""
        binned_by_stratum = {}
        for stratum, results in results_by_stratum.items():
            all_vals = []
            for r in results:
                for trace in r["traces"]:
                    all_vals.append(trace[field])
            if not all_vals:
                continue

            max_steps = max(len(v) for v in all_vals)
            n_bins = (max_steps + bin_size - 1) // bin_size
            binned = np.full((len(all_vals), n_bins), np.nan)
            for i, vals in enumerate(all_vals):
                arr = np.array(vals, dtype=np.float64)
                for b in range(n_bins):
                    start = b * bin_size
                    end = min(start + bin_size, len(arr))
                    if start < len(arr):
                        binned[i, b] = np.mean(np.abs(arr[start:end]))
            binned_by_stratum[stratum] = binned
        return binned_by_stratum

    # ── F0: |Δw| vs step, side by side ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, variant in zip(axes, variants):
        binned_data = _collect_binned(all_results[variant], "dw", bin_size)
        for stratum in TARGET_STRATA:
            if stratum not in binned_data:
                continue
            binned = binned_data[stratum]
            n_bins = binned.shape[1]
            bin_centers = (np.arange(n_bins) * bin_size) + bin_size / 2
            means = np.nanmean(binned, axis=0)
            counts = np.sum(~np.isnan(binned), axis=0)
            sems = np.nanstd(binned, axis=0) / np.sqrt(np.maximum(counts, 1))

            color = stratum_colors[stratum]
            params = VARIANT_PARAMS[variant][stratum]
            label = f"{stratum_labels[stratum]} (η={params['eta']})"
            ax.plot(bin_centers, means, color=color, label=label, linewidth=2)
            ax.fill_between(bin_centers, means - sems, means + sems, color=color, alpha=0.15)

        ax.axvline(x=SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label=f"Physics switch (step {SWITCH_STEP})")
        ax.set_xlabel(f"Step ({bin_size}-step bins)", fontsize=11)
        ax.set_title(variant_labels[variant], fontsize=13)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean |Δw| per step", fontsize=11)
    fig.suptitle("Temporal Profile of Weight Changes Under Non-Stationary CartPole", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "F0_dw_vs_step_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved F0: F0_dw_vs_step_comparison.png")

    # ── F1: |θ| vs step, side by side ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, variant in zip(axes, variants):
        binned_data = _collect_binned(all_results[variant], "theta", bin_size)
        for stratum in TARGET_STRATA:
            if stratum not in binned_data:
                continue
            binned = binned_data[stratum]
            n_bins = binned.shape[1]
            bin_centers = (np.arange(n_bins) * bin_size) + bin_size / 2
            means = np.nanmean(binned, axis=0)
            counts = np.sum(~np.isnan(binned), axis=0)
            sems = np.nanstd(binned, axis=0) / np.sqrt(np.maximum(counts, 1))

            color = stratum_colors[stratum]
            ax.plot(bin_centers, means, color=color, label=stratum_labels[stratum], linewidth=2)
            ax.fill_between(bin_centers, means - sems, means + sems, color=color, alpha=0.15)

        ax.axvline(x=SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label=f"Physics switch")
        ax.set_xlabel(f"Step ({bin_size}-step bins)", fontsize=11)
        ax.set_title(variant_labels[variant], fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean |θ| (pole angle, rad)", fontsize=11)
    fig.suptitle("Pole Angle Under Non-Stationary CartPole", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "F1_theta_vs_step_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved F1: F1_theta_vs_step_comparison.png")

    # ── F2: |Δw| + |θ| overlay for High-mid ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, variant in zip(axes, variants):
        hm_results = all_results[variant].get("high_mid", [])
        if not hm_results:
            continue

        # Collect all High-mid traces
        all_dw = []
        all_theta = []
        for r in hm_results:
            for trace in r["traces"]:
                all_dw.append(trace["dw"])
                all_theta.append(trace["theta"])

        if not all_dw:
            continue

        max_steps = max(len(t) for t in all_dw)
        n_bins = (max_steps + bin_size - 1) // bin_size
        bin_centers = (np.arange(n_bins) * bin_size) + bin_size / 2

        # Bin |Δw|
        dw_binned = np.full((len(all_dw), n_bins), np.nan)
        for i, t in enumerate(all_dw):
            arr = np.array(t, dtype=np.float64)
            for b in range(n_bins):
                s, e = b * bin_size, min((b + 1) * bin_size, len(arr))
                if s < len(arr):
                    dw_binned[i, b] = arr[s:e].mean()

        # Bin |θ|
        theta_binned = np.full((len(all_theta), n_bins), np.nan)
        for i, t in enumerate(all_theta):
            arr = np.abs(np.array(t, dtype=np.float64))
            for b in range(n_bins):
                s, e = b * bin_size, min((b + 1) * bin_size, len(arr))
                if s < len(arr):
                    theta_binned[i, b] = arr[s:e].mean()

        dw_means = np.nanmean(dw_binned, axis=0)
        theta_means = np.nanmean(theta_binned, axis=0)

        ax.plot(bin_centers, dw_means, color="#FF9800", label="|Δw|", linewidth=2)
        ax.set_ylabel("|Δw|", color="#FF9800", fontsize=11)
        ax.tick_params(axis="y", labelcolor="#FF9800")

        ax2 = ax.twinx()
        ax2.plot(bin_centers, theta_means, color="#E91E63", label="|θ|", linewidth=2, linestyle="-.")
        ax2.set_ylabel("|θ| (rad)", color="#E91E63", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#E91E63")

        ax.axvline(x=SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        params = VARIANT_PARAMS[variant]["high_mid"]
        ax.set_title(f"{variant_labels[variant]}\n(η={params['eta']})", fontsize=12)
        ax.set_xlabel(f"Step ({bin_size}-step bins)", fontsize=11)
        ax.grid(True, alpha=0.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle("High-mid: Weight Changes vs Pole Angle", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "F2_dw_theta_overlay_high_mid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved F2: F2_dw_theta_overlay_high_mid.png")

    # ── F3: x_dot response — control authority comparison ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, variant in zip(axes, variants):
        binned_data = _collect_binned(all_results[variant], "x_dot", bin_size)
        for stratum in TARGET_STRATA:
            if stratum not in binned_data:
                continue
            binned = binned_data[stratum]
            n_bins = binned.shape[1]
            bin_centers = (np.arange(n_bins) * bin_size) + bin_size / 2
            means = np.nanmean(binned, axis=0)

            color = stratum_colors[stratum]
            ax.plot(bin_centers, means, color=color, label=stratum_labels[stratum], linewidth=2)

        ax.axvline(x=SWITCH_STEP, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label="Physics switch")
        ax.set_xlabel(f"Step ({bin_size}-step bins)", fontsize=11)
        ax.set_title(variant_labels[variant], fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean |ẋ| (cart velocity)", fontsize=11)
    fig.suptitle("Cart Velocity — Control Authority Proxy", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "F3_x_dot_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved F3: F3_x_dot_comparison.png")

    # ── F4: P1 vs P2 |Δw| ratio per stratum ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(TARGET_STRATA))
    width = 0.35

    for vi, variant in enumerate(variants):
        p1_means = []
        p2_means = []
        for stratum in TARGET_STRATA:
            results = all_results[variant].get(stratum, [])
            p1_vals = []
            p2_vals = []
            for r in results:
                for trace in r["traces"]:
                    dw_arr = np.array(trace["dw"])
                    phase_arr = np.array(trace["phase"])
                    p1_mask = phase_arr == 1
                    p2_mask = phase_arr == 2
                    if p1_mask.any():
                        p1_vals.append(float(dw_arr[p1_mask].mean()))
                    if p2_mask.any():
                        p2_vals.append(float(dw_arr[p2_mask].mean()))
            p1_means.append(np.mean(p1_vals) if p1_vals else 0)
            p2_means.append(np.mean(p2_vals) if p2_vals else 0)

        # Compute P2/P1 ratio
        ratios = []
        for p1, p2 in zip(p1_means, p2_means):
            ratios.append(p2 / p1 if p1 > 1e-15 else 0)

        offset = (vi - 0.5) * width
        bars = ax.bar(x_pos + offset, ratios, width, label=variant_labels[variant], alpha=0.8)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="P2/P1 = 1.0 (no change)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([stratum_labels[s] for s in TARGET_STRATA], fontsize=11)
    ax.set_ylabel("P2 / P1 mean |Δw| ratio", fontsize=11)
    ax.set_title("Does Plasticity Increase After the Physics Switch?", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "F4_p2_p1_dw_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  Saved F4: F4_p2_p1_dw_ratio.png")


def main():
    parser = argparse.ArgumentParser(
        description="B0.6 Non-Stationary Temporal |Δw| + Physics Profiling"
    )
    parser.add_argument(
        "--variant", default="both",
        choices=["gravity_2x", "heavy_pole", "both"],
        help="Which variant(s) to profile (default: both)",
    )
    parser.add_argument(
        "--per-stratum", type=int, default=50,
        help="Networks per stratum (default: 50)",
    )
    parser.add_argument(
        "--rollouts", type=int, default=20,
        help="Episodes per network (default: 20)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Parallel workers (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--bin-size", type=int, default=10,
        help="Bin width for plots (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    n_workers = args.n_workers or max(1, os.cpu_count() - 2)
    variants = ["gravity_2x", "heavy_pole"] if args.variant == "both" else [args.variant]

    # Load and sample networks (same sample for both variants)
    print("Loading networks...")
    sampled = _load_networks_by_stratum(args.per_stratum, seed=args.seed)
    total_networks = sum(len(ids) for ids in sampled.values())

    # Build work items for all variants
    work_items: list[tuple] = []
    for variant in variants:
        for stratum, ids in sampled.items():
            params = VARIANT_PARAMS[variant][stratum]
            for nid in ids:
                fpath = B05P_POOL / f"network_{nid:05d}.json"
                work_items.append((
                    str(fpath), nid, stratum, variant,
                    params["eta"], params["decay"], args.rollouts,
                ))

    total_evals = len(work_items) * args.rollouts
    print(f"\n{'='*70}")
    print(f"B0.6 Non-Stationary Temporal Profiling")
    print(f"  Variants: {variants}")
    print(f"  Networks: {total_networks} ({args.per_stratum} × {len(sampled)} strata)")
    print(f"  Rollouts: {args.rollouts} per network per variant")
    print(f"  Work items: {len(work_items):,}  |  Total evaluations: {total_evals:,}")
    print(f"  Workers: {n_workers}")
    print(f"  Switch step: {SWITCH_STEP}")
    for variant in variants:
        print(f"  {variant} params:")
        for s in TARGET_STRATA:
            p = VARIANT_PARAMS[variant][s]
            print(f"    {s}: η={p['eta']}, decay={p['decay']}")
    print(f"{'='*70}\n")

    # Run evaluations
    shutdown_event = mp.Event()
    # Structure: variant → stratum → list[result]
    all_results: dict[str, dict[str, list[dict]]] = {
        v: {s: [] for s in TARGET_STRATA} for v in variants
    }
    t0 = time.time()

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shutdown_event,),
        ) as pool:
            total = len(work_items)
            for i, result in enumerate(pool.imap_unordered(_evaluate_single, work_items, chunksize=2)):
                if result is not None:
                    all_results[result["variant"]][result["stratum"]].append(result)

                if (i + 1) % 20 == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta_remaining = (total - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1}/{total}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"{rate:.1f} net/s, "
                        f"~{eta_remaining:.0f}s remaining"
                    )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        shutdown_event.set()

    elapsed = time.time() - t0
    collected = sum(
        len(rs) for v_results in all_results.values() for rs in v_results.values()
    )
    print(f"\nCollected {collected} network profiles in {elapsed:.1f}s")

    if collected == 0:
        print("No results. Exiting.")
        return

    # Save traces
    print("\nSaving traces...")
    _save_traces(all_results, OUTPUT_ROOT)

    # Save metadata
    metadata_rows: list[dict] = []
    for variant, stratum_results in all_results.items():
        for stratum, results in stratum_results.items():
            for r in results:
                p1_dw = []
                p2_dw = []
                for trace in r["traces"]:
                    dw_arr = np.array(trace["dw"])
                    phase_arr = np.array(trace["phase"])
                    p1_mask = phase_arr == 1
                    p2_mask = phase_arr == 2
                    if p1_mask.any():
                        p1_dw.append(float(dw_arr[p1_mask].mean()))
                    if p2_mask.any():
                        p2_dw.append(float(dw_arr[p2_mask].mean()))

                metadata_rows.append({
                    "network_id": r["network_id"],
                    "stratum": r["stratum"],
                    "variant": r["variant"],
                    "eta": r["eta"],
                    "decay": r["decay"],
                    "n_episodes": len(r["traces"]),
                    "avg_reward": float(np.mean(r["rewards"])),
                    "avg_length": float(np.mean(r["lengths"])),
                    "p1_mean_dw": float(np.mean(p1_dw)) if p1_dw else 0,
                    "p2_mean_dw": float(np.mean(p2_dw)) if p2_dw else 0,
                    "p2_p1_ratio": (float(np.mean(p2_dw)) / float(np.mean(p1_dw)))
                        if p1_dw and p2_dw and np.mean(p1_dw) > 1e-15 else 0,
                })

    with open(OUTPUT_ROOT / "metadata.json", "w") as f:
        json.dump({
            "experiment": "B0.6 Non-Stationary Temporal Profiling",
            "variants": variants,
            "variant_params": {v: VARIANT_PARAMS[v] for v in variants},
            "per_stratum": args.per_stratum,
            "rollouts": args.rollouts,
            "switch_step": SWITCH_STEP,
            "seed": args.seed,
            "bin_size": args.bin_size,
            "elapsed_seconds": round(elapsed, 1),
            "networks": metadata_rows,
        }, f, indent=2)
    print(f"Saved metadata: {OUTPUT_ROOT / 'metadata.json'}")

    # Generate plots
    print("\nGenerating plots...")
    _plot_comparison(all_results, OUTPUT_ROOT, bin_size=args.bin_size)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary: P1 vs P2 Mean |Δw|")
    print(f"{'='*70}")
    print(f"{'Variant':<14} {'Stratum':<15} {'Nets':>5} {'AvgLen':>7} {'P1 |Δw|':>10} {'P2 |Δw|':>10} {'P2/P1':>7}")
    print("-" * 70)

    for variant in variants:
        for stratum in TARGET_STRATA:
            rows = [m for m in metadata_rows
                    if m["variant"] == variant and m["stratum"] == stratum]
            if not rows:
                continue
            n = len(rows)
            avg_len = np.mean([r["avg_length"] for r in rows])
            p1 = np.mean([r["p1_mean_dw"] for r in rows])
            p2 = np.mean([r["p2_mean_dw"] for r in rows])
            ratio = p2 / p1 if p1 > 1e-15 else 0
            print(f"{variant:<14} {stratum:<15} {n:>5} {avg_len:>7.1f} {p1:>10.7f} {p2:>10.7f} {ratio:>7.2f}")
        print()

    print(f"{'='*70}")
    print(f"All outputs in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
