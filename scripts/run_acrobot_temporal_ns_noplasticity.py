#!/usr/bin/env python3
"""
Acrobot Non-Stationary No-Plasticity Temporal Trace Runner

Generates per-timestep traces for the same networks used in the NS temporal
profiling, under the same non-stationary conditions (heavy_link2_2x, switch
at step 50) but with plasticity disabled (η=0, decay=0).

Serves as a no-plasticity baseline: the environment changes, the network
cannot adapt, and we capture the behavioral impact via observation traces.

Usage:
  python scripts/run_acrobot_temporal_ns_noplasticity.py
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

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome
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

logger = logging.getLogger(__name__)

POOL_NETWORKS = Path("experiments/acrobot/pool/networks")
DEFAULT_METADATA = Path("experiments/acrobot/temporal_profile/metadata_acrobot.json")
DEFAULT_OUTPUT_DIR = Path("experiments/acrobot/temporal_profile")

VARIANT = "heavy_link2_2x"


def _evaluate_ns_noplasticity(args: tuple) -> dict | None:
    """Evaluate one network on NS Acrobot with no plasticity hook."""
    network_path, switch_step, rollouts = args
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

        target_params = VARIANTS[VARIANT]
        base_env = gym.make(ENV_NAME, render_mode=None)
        env = AcrobotMidEpisodeSwitchWrapper(
            base_env, switch_step=switch_step, target_params=target_params,
        )

        seeds = VERIFICATION_SEEDS[:int(rollouts)]

        try:
            # No plasticity hook — edge_hook=None
            prop = create_propagator(grid, edge_hook=None, graph_diameter=graph_diameter)
            traces = run_rollouts_temporal(
                prop, int(rollouts),
                seeds=seeds, reset_plastic_each_episode=True,
                env=env, switch_step=switch_step,
            )
        finally:
            env.close()

        rewards = [t["total_reward"] for t in traces]
        dws = [float(t["dw"].mean()) for t in traces]

        return {
            "traces": traces,
            "metadata": {
                "network_id": int(network_id),
                "stratum": stratum,
                "variant": VARIANT,
                "eta": 0.0,
                "decay": 0.0,
                "switch_step": switch_step,
                "n_episodes": len(traces),
                "avg_reward": float(np.mean(rewards)),
                "mean_dw": float(np.mean(dws)),
                "num_neurons": num_nodes,
                "num_connections": num_edges,
            },
        }
    except Exception as e:
        logger.error("Error evaluating %s: %s", network_path, e)
        return None


def _worker_init(shutdown_event):
    set_shutdown_event(shutdown_event)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from MorphoNAS_PL.parallel_utils import configure_worker_threads
    configure_worker_threads()


def main():
    parser = argparse.ArgumentParser(
        description="Acrobot NS No-Plasticity Temporal Traces",
    )
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA,
        help="Path to NS metadata JSON (for network selection)",
    )
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    n_workers = args.n_workers or max(1, os.cpu_count() - 2)

    with open(args.metadata) as f:
        ns_meta = json.load(f)

    networks = ns_meta["networks"]
    switch_step = ns_meta.get("switch_step", 50)

    print(f"\n{'='*60}")
    print("ACROBOT NS NO-PLASTICITY TEMPORAL TRACES (baseline)")
    print(f"{'='*60}")
    print(f"  Networks: {len(networks)} (from {args.metadata})")
    print(f"  Variant: {VARIANT} (switch at step {switch_step})")
    print(f"  Plasticity: OFF (η=0, decay=0)")
    print(f"  Rollouts: {args.rollouts}")
    print(f"  Workers: {n_workers}")

    work_items: list[tuple] = []
    for net in networks:
        nid = net["network_id"]
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            logger.warning("Network file not found: %s", fpath)
            continue
        work_items.append((fpath, switch_step, args.rollouts))

    print(f"  Work items: {len(work_items)}")
    print()

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
                pool.imap_unordered(_evaluate_ns_noplasticity, work_items, chunksize=1)
            ):
                if result is None:
                    continue
                results.append(result)

                if (i + 1) % 10 == 0 or (i + 1) == len(work_items):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(work_items) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{i+1}/{len(work_items)}] "
                        f"{elapsed:.0f}s elapsed, "
                        f"~{remaining:.0f}s remaining"
                    )

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        shutdown_event.set()

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} network evaluations in {elapsed:.1f}s")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    npz_path = args.output_dir / "traces_ns_no_plasticity.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    print(f"  Saved {len(npz_arrays)} arrays to {npz_path}")

    meta_doc = {
        "experiment": "Acrobot NS No-Plasticity Temporal Profiling (baseline)",
        "variant": VARIANT,
        "variant_params": VARIANTS[VARIANT],
        "per_stratum": ns_meta.get("per_stratum", 50),
        "rollouts": args.rollouts,
        "switch_step": switch_step,
        "plasticity": False,
        "seed": ns_meta.get("seed", 42),
        "n_networks": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "networks": metadata_list,
    }
    meta_path = args.output_dir / "metadata_acrobot_ns_no_plasticity.json"
    with open(meta_path, "w") as f:
        json.dump(meta_doc, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    # Summary
    strata_order = ["low_mid", "high_mid", "near_perfect", "perfect"]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in strata_order:
        s_results = [r for r in results if r["metadata"]["stratum"] == s]
        if not s_results:
            continue
        rewards = [r["metadata"]["avg_reward"] for r in s_results]
        dws = [r["metadata"]["mean_dw"] for r in s_results]
        print(f"  {s} ({len(s_results)} nets):")
        print(f"    Mean reward: {np.mean(rewards):.1f}")
        print(f"    Mean |Δw|:   {np.mean(dws):.8f}  (should be ~0)")

    print(f"\nAll done! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
