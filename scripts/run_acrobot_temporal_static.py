#!/usr/bin/env python3
"""
Acrobot Static Temporal Trace Runner

Generates per-timestep traces for the same networks used in the non-stationary
temporal profiling, but with no physics switch (standard Acrobot-v1 throughout).
Serves as a control to compare against the non-stationary traces.

Reads network selection and hyperparameters from the existing NS metadata file.

Usage:
  python scripts/run_acrobot_temporal_static.py
  python scripts/run_acrobot_temporal_static.py --metadata experiments/acrobot/temporal_profile/metadata_acrobot.json
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
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_PL.experiment_acrobot import (
    ENV_NAME,
    VERIFICATION_SEEDS,
    load_network_from_file,
)
from MorphoNAS_PL.experiment_acrobot_temporal import (
    run_rollouts_temporal,
    set_shutdown_event,
)
from MorphoNAS_PL.experimentB0_5_natural import create_propagator
from MorphoNAS_PL.experiment_acrobot import create_propagator
from MorphoNAS_PL.plasticity_hooks import HebbianEdgeHook

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid

import networkx as nx

logger = logging.getLogger(__name__)

POOL_NETWORKS = Path("experiments/acrobot/pool/networks")
DEFAULT_METADATA = Path("experiments/acrobot/temporal_profile/metadata_acrobot.json")
DEFAULT_OUTPUT_DIR = Path("experiments/acrobot/temporal_profile")


def _evaluate_static(args: tuple) -> dict | None:
    """Evaluate one network on static Acrobot with temporal recording."""
    network_path, eta, decay, rollouts = args
    try:
        genome, metadata = load_network_from_file(str(network_path))
        network_id = metadata.get("network_id")
        stratum = metadata.get("stratum")

        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        G = grid.get_graph()
        num_nodes = int(G.number_of_nodes())
        num_edges = int(G.number_of_edges())

        if num_nodes < 9:  # INPUT_DIM + OUTPUT_DIM
            return None

        try:
            graph_diameter = int(nx.diameter(G))
        except nx.NetworkXError:
            max_length = 0
            for source in G.nodes():
                lengths = nx.shortest_path_length(G, source)
                max_length = max(max_length, max(lengths.values()))
            graph_diameter = int(max_length)

        # Plain Acrobot — no switch wrapper
        env = gym.make(ENV_NAME, render_mode=None)
        seeds = VERIFICATION_SEEDS[:int(rollouts)]

        try:
            hook = HebbianEdgeHook(learning_rate=float(eta), weight_decay=float(decay))
            prop = create_propagator(grid, edge_hook=hook, graph_diameter=graph_diameter)
            traces = run_rollouts_temporal(
                prop, int(rollouts),
                seeds=seeds, reset_plastic_each_episode=True,
                env=env, switch_step=99999,  # never triggers
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
                "variant": "static",
                "eta": float(eta),
                "decay": float(decay),
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
        description="Acrobot Static Temporal Trace Runner",
    )
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA,
        help="Path to NS metadata JSON (for network selection + hyperparams)",
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

    # Read NS metadata to get same networks and hyperparams
    with open(args.metadata) as f:
        ns_meta = json.load(f)

    networks = ns_meta["networks"]
    print(f"\n{'='*60}")
    print("ACROBOT STATIC TEMPORAL TRACES (control)")
    print(f"{'='*60}")
    print(f"  Networks: {len(networks)} (from {args.metadata})")
    print(f"  Rollouts: {args.rollouts}")
    print(f"  Workers: {n_workers}")
    print(f"  No physics switch (standard Acrobot-v1)")

    # Build work items from metadata
    work_items: list[tuple] = []
    for net in networks:
        nid = net["network_id"]
        fpath = POOL_NETWORKS / f"network_{nid:05d}.json"
        if not fpath.exists():
            logger.warning("Network file not found: %s", fpath)
            continue
        work_items.append((fpath, net["eta"], net["decay"], args.rollouts))

    print(f"  Work items: {len(work_items)}")
    print()

    # Run
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
                pool.imap_unordered(_evaluate_static, work_items, chunksize=1)
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

    # Save NPZ + metadata
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

    npz_path = args.output_dir / "traces_static.npz"
    np.savez_compressed(npz_path, **npz_arrays)
    print(f"  Saved {len(npz_arrays)} arrays to {npz_path}")

    meta_doc = {
        "experiment": "Acrobot Static Temporal Profiling (control)",
        "variant": "static",
        "variant_params": ns_meta.get("variant_params", {}),
        "per_stratum": ns_meta.get("per_stratum", 50),
        "rollouts": args.rollouts,
        "switch_step": None,
        "seed": ns_meta.get("seed", 42),
        "n_networks": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "networks": metadata_list,
    }
    meta_path = args.output_dir / "metadata_acrobot_static.json"
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
        dws = [r["metadata"]["mean_dw"] for r in s_results]
        print(f"  {s} ({len(s_results)} nets): mean |Δw| = {np.mean(dws):.6f}")

    print(f"\nAll done! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
