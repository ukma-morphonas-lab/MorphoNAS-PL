#!/usr/bin/env python3
"""
Acrobot Temporal Traces with Per-Network Oracle Hyperparameters

Runs temporal traces using each network's individually optimal (η, λ) rather
than a fixed per-stratum setting. Networks where plasticity doesn't help
(oracle delta ≤ 0) get η=0.

Produces two conditions:
  1. NS + oracle plasticity  → traces_heavy_link2_2x.npz
  2. Static + oracle plasticity → traces_static.npz

The NS no-plasticity baseline (traces_ns_no_plasticity.npz) is unchanged.

Usage:
  python scripts/run_acrobot_temporal_oracle.py
  python scripts/run_acrobot_temporal_oracle.py --per-stratum 5  # quick test
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

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("code"))

from MorphoNAS.grid import Grid
from MorphoNAS_PL.env_wrappers_acrobot import AcrobotMidEpisodeSwitchWrapper
from MorphoNAS_PL.experiment_acrobot import (
    ENV_NAME,
    VERIFICATION_SEEDS,
    create_propagator,
    load_network_from_file,
)
from MorphoNAS_PL.experiment_acrobot_nonstationary import VARIANTS
from MorphoNAS_PL.experiment_acrobot_temporal import (
    run_rollouts_temporal,
    set_shutdown_event,
)
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

logger = logging.getLogger(__name__)

POOL_NETWORKS = Path("experiments/acrobot/pool/networks")
NS_SWEEP_PATH = Path(
    "experiments/acrobot/sweep_nonstationary/sweep/acrobot_ns_heavy_link2_2x.parquet"
)
STATIC_SWEEP_PATH = Path("experiments/acrobot/sweep_static/acrobot_static_sweep.parquet")
OUTPUT_DIR = Path("experiments/acrobot/temporal_profile")

VARIANT = "heavy_link2_2x"
NS_SWITCH_STEP = 50
NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


# ── Oracle computation ───────────────────────────────────────────────

def compute_per_network_oracle(
    sweep_df: pd.DataFrame,
    delta_col: str,
) -> pd.DataFrame:
    """Find per-network oracle (η, λ) that maximizes delta_col.

    Returns DataFrame with columns: network_id, stratum, oracle_eta,
    oracle_decay, oracle_delta, use_plasticity.
    """
    idx = sweep_df.groupby("network_id")[delta_col].idxmax()
    oracle = sweep_df.loc[idx, ["network_id", "stratum", "eta", "decay", delta_col]].copy()
    oracle = oracle.rename(columns={
        "eta": "oracle_eta",
        "decay": "oracle_decay",
        delta_col: "oracle_delta",
    })
    # Networks where plasticity doesn't help get η=0
    oracle["use_plasticity"] = oracle["oracle_delta"] > 0
    oracle.loc[~oracle["use_plasticity"], "oracle_eta"] = 0.0
    oracle.loc[~oracle["use_plasticity"], "oracle_decay"] = 0.0
    return oracle.reset_index(drop=True)


def select_networks(
    oracle_df: pd.DataFrame,
    per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified selection of networks, returning oracle params for each."""
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []

    for stratum in NON_WEAK_STRATA:
        sdf = oracle_df[oracle_df["stratum"] == stratum]
        n = min(per_stratum, len(sdf))
        if n == 0:
            continue
        chosen_idx = rng.choice(len(sdf), size=n, replace=False)
        chosen = sdf.iloc[chosen_idx]
        selected.append(chosen)
        logger.info("  %s: %d networks selected", stratum, n)

    return pd.concat(selected, ignore_index=True)


# ── Worker functions ─────────────────────────────────────────────────

def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def _evaluate_temporal(args: tuple) -> dict | None:
    """Evaluate one network with given (η, λ) and env config."""
    network_path, eta, decay, use_ns, switch_step, rollouts = args
    try:
        genome, metadata = load_network_from_file(str(network_path))
        network_id = metadata.get("network_id")
        stratum = metadata.get("stratum")

        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        G = grid.get_graph()
        num_nodes = int(G.number_of_nodes())
        num_edges = int(G.number_of_edges())

        if num_nodes < 9:
            return None

        try:
            graph_diameter = int(nx.diameter(G))
        except nx.NetworkXError:
            max_length = 0
            for source in G.nodes():
                lengths = nx.shortest_path_length(G, source)
                max_length = max(max_length, max(lengths.values()))
            graph_diameter = int(max_length)

        # Create env
        base_env = gym.make(ENV_NAME, render_mode=None)
        if use_ns:
            target_params = VARIANTS[VARIANT]
            env = AcrobotMidEpisodeSwitchWrapper(
                base_env, switch_step=switch_step, target_params=target_params,
            )
        else:
            env = base_env

        seeds = VERIFICATION_SEEDS[:int(rollouts)]

        try:
            if abs(eta) > 1e-12:
                hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))
                prop = create_propagator(grid, edge_hook=hook, graph_diameter=graph_diameter)
            else:
                prop = create_propagator(grid, edge_hook=None, graph_diameter=graph_diameter)

            traces = run_rollouts_temporal(
                prop, int(rollouts),
                seeds=seeds, reset_plastic_each_episode=True,
                env=env, switch_step=switch_step if use_ns else 99999,
            )
        finally:
            env.close()

        rewards = [t["total_reward"] for t in traces]
        p1_dws, p2_dws = [], []
        for t in traces:
            mask1 = t["phase"] == 1
            mask2 = t["phase"] == 2
            if mask1.any():
                p1_dws.append(float(t["dw"][mask1].mean()))
            if mask2.any():
                p2_dws.append(float(t["dw"][mask2].mean()))

        return {
            "traces": traces,
            "metadata": {
                "network_id": int(network_id),
                "stratum": stratum,
                "eta": float(eta),
                "decay": float(decay),
                "use_plasticity": abs(eta) > 1e-12,
                "n_episodes": len(traces),
                "avg_reward": float(np.mean(rewards)),
                "mean_episode_length": float(np.mean([t["total_steps"] for t in traces])),
                "p1_mean_dw": float(np.mean(p1_dws)) if p1_dws else 0.0,
                "p2_mean_dw": float(np.mean(p2_dws)) if p2_dws else 0.0,
                "p2_p1_ratio": (
                    float(np.mean(p2_dws)) / float(np.mean(p1_dws))
                    if p1_dws and np.mean(p1_dws) > 1e-15 else 0.0
                ),
                "num_neurons": num_nodes,
                "num_connections": num_edges,
            },
        }
    except Exception as e:
        logger.error("Error evaluating %s: %s", network_path, e)
        return None


# ── Run + save ───────────────────────────────────────────────────────

def run_condition(
    selection: pd.DataFrame,
    use_ns: bool,
    switch_step: int,
    rollouts: int,
    n_workers: int,
    label: str,
) -> list[dict]:
    """Run one condition (NS or static) and return results."""
    work_items = []
    for _, row in selection.iterrows():
        nid = int(row["network_id"])
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            continue
        work_items.append((
            fpath, float(row["oracle_eta"]), float(row["oracle_decay"]),
            use_ns, switch_step, rollouts,
        ))

    print(f"\n  --- {label} ({len(work_items)} networks) ---")

    shutdown_event = mp.Event()
    results: list[dict] = []
    t0 = time.time()

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shutdown_event,),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_evaluate_temporal, work_items, chunksize=1)
            ):
                if result is None:
                    continue
                results.append(result)

                if (i + 1) % 10 == 0 or (i + 1) == len(work_items):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(work_items) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"    [{i+1}/{len(work_items)}] "
                        f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
                    )

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"  Completed {len(results)} evaluations in {elapsed:.1f}s")
    return results


def save_results(
    results: list[dict],
    npz_path: Path,
    meta_path: Path,
    experiment_label: str,
    variant: str,
    switch_step: int | None,
    seed: int,
    per_stratum: int,
    rollouts: int,
    elapsed: float,
) -> None:
    """Save NPZ and metadata JSON."""
    npz_arrays: dict[str, np.ndarray] = {}
    metadata_list: list[dict] = []

    for result in results:
        meta = result["metadata"]
        stratum = meta["stratum"]
        nid = meta["network_id"]
        metadata_list.append(meta)

        for ep_idx, trace in enumerate(result["traces"]):
            prefix = f"{stratum}_{nid}_{ep_idx}"
            npz_arrays[f"{prefix}_dw"] = trace["dw"]
            npz_arrays[f"{prefix}_obs"] = trace["obs"]
            npz_arrays[f"{prefix}_action"] = trace["action"]
            npz_arrays[f"{prefix}_phase"] = trace["phase"]

    np.savez_compressed(npz_path, **npz_arrays)
    print(f"  Saved {len(npz_arrays)} arrays to {npz_path}")

    meta_doc = {
        "experiment": experiment_label,
        "variant": variant,
        "hyperparams": "per_network_oracle",
        "per_stratum": per_stratum,
        "rollouts": rollouts,
        "switch_step": switch_step,
        "seed": seed,
        "n_networks": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "networks": metadata_list,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_doc, f, indent=2)
    print(f"  Saved metadata to {meta_path}")


def print_summary(
    results: list[dict],
    label: str,
    show_dw_ratio: bool = False,
) -> None:
    """Print per-stratum summary."""
    print(f"\n  {label}:")
    print(f"  {'Stratum':<16s} {'n':>4s} {'η>0':>5s} {'η<0':>5s} {'η=0':>5s} "
          f"{'MeanLen':>8s}", end="")
    if show_dw_ratio:
        print(f" {'P2/P1 dw':>10s}", end="")
    print()
    print("  " + "-" * (50 + (12 if show_dw_ratio else 0)))

    for s in NON_WEAK_STRATA:
        s_results = [r for r in results if r["metadata"]["stratum"] == s]
        if not s_results:
            continue
        etas = [r["metadata"]["eta"] for r in s_results]
        n_pos = sum(1 for e in etas if e > 1e-12)
        n_neg = sum(1 for e in etas if e < -1e-12)
        n_zero = len(etas) - n_pos - n_neg
        mean_len = np.mean([r["metadata"]["mean_episode_length"] for r in s_results])

        line = f"  {s:<16s} {len(s_results):>4d} {n_pos:>5d} {n_neg:>5d} {n_zero:>5d} {mean_len:>8.1f}"

        if show_dw_ratio:
            ratios = [r["metadata"]["p2_p1_ratio"] for r in s_results
                      if r["metadata"]["p2_p1_ratio"] > 0]
            ratio_str = f"{np.median(ratios):.1f}" if ratios else "N/A"
            line += f" {ratio_str:>10s}"

        print(line)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Acrobot Temporal Traces with Per-Network Oracle Hyperparameters",
    )
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    n_workers = args.n_workers or max(1, os.cpu_count() - 2)

    print(f"\n{'='*60}")
    print("ACROBOT TEMPORAL TRACES — PER-NETWORK ORACLE")
    print(f"{'='*60}")

    # Step 1: Compute per-network oracles
    print("\nStep 1: Computing per-network oracle (η, λ)...")

    ns_sweep = pd.read_parquet(NS_SWEEP_PATH)
    ns_sweep = ns_sweep[ns_sweep["stratum"].isin(NON_WEAK_STRATA)]
    ns_oracle = compute_per_network_oracle(ns_sweep, "delta_reward_total")
    n_helped_ns = ns_oracle["use_plasticity"].sum()
    print(f"  NS oracle: {n_helped_ns}/{len(ns_oracle)} networks benefit from plasticity")

    static_sweep = pd.read_parquet(STATIC_SWEEP_PATH)
    static_sweep = static_sweep[static_sweep["stratum"].isin(NON_WEAK_STRATA)]
    static_oracle = compute_per_network_oracle(static_sweep, "delta_reward")
    n_helped_st = static_oracle["use_plasticity"].sum()
    print(f"  Static oracle: {n_helped_st}/{len(static_oracle)} networks benefit")

    # Step 2: Select same 200 networks for both conditions
    print(f"\nStep 2: Selecting {args.per_stratum} networks per stratum (seed={args.seed})...")
    ns_selection = select_networks(ns_oracle, args.per_stratum, args.seed)
    print(f"  Selected {len(ns_selection)} networks (NS)")

    # For static, use same network_ids but with static oracle params
    selected_ids = set(ns_selection["network_id"].values)
    static_selection = static_oracle[static_oracle["network_id"].isin(selected_ids)].copy()
    print(f"  Matched {len(static_selection)} networks (static)")

    # Step 3: Run NS + oracle plasticity
    print(f"\nStep 3: Running temporal traces...")
    t0_total = time.time()

    ns_results = run_condition(
        ns_selection, use_ns=True, switch_step=NS_SWITCH_STEP,
        rollouts=args.rollouts, n_workers=n_workers,
        label="NS + oracle plasticity",
    )

    # Step 4: Run Static + oracle plasticity
    static_results = run_condition(
        static_selection, use_ns=False, switch_step=NS_SWITCH_STEP,
        rollouts=args.rollouts, n_workers=n_workers,
        label="Static + oracle plasticity",
    )

    total_elapsed = time.time() - t0_total

    # Step 5: Save
    print(f"\nStep 4: Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_results(
        ns_results,
        OUTPUT_DIR / "traces_heavy_link2_2x.npz",
        OUTPUT_DIR / "metadata_acrobot.json",
        "Acrobot NS Temporal Profiling (per-network oracle)",
        VARIANT, NS_SWITCH_STEP, args.seed, args.per_stratum,
        args.rollouts, total_elapsed,
    )

    save_results(
        static_results,
        OUTPUT_DIR / "traces_static.npz",
        OUTPUT_DIR / "metadata_acrobot_static.json",
        "Acrobot Static Temporal Profiling (per-network oracle)",
        "static", None, args.seed, args.per_stratum,
        args.rollouts, total_elapsed,
    )

    # Step 6: Summary
    print(f"\n{'='*60}")
    print("PER-STRATUM SUMMARY")
    print(f"{'='*60}")
    print_summary(ns_results, "NS + oracle plasticity", show_dw_ratio=True)
    print_summary(static_results, "Static + oracle plasticity", show_dw_ratio=False)

    print(f"\nTotal time: {total_elapsed:.0f}s")
    print(f"All done! Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
