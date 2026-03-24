#!/usr/bin/env python3
"""
Extract pool_natural.parquet back to individual network JSON files.

Needed only if you want to re-run Experiment 1 Stage 1 (B0.5) or
Stage 2 (B0.5+) from scratch. The 2 862-network subsample required
for all paper experiments is already committed under
experiments/B0.5+/pool_subsample/networks/.

Usage:
    python scripts/extract_pool.py
    python scripts/extract_pool.py --parquet experiments/B0.5/pool_natural/pool_natural.parquet \
                                   --out-dir  experiments/B0.5/pool_natural/networks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack pool parquet → JSON files")
    parser.add_argument(
        "--parquet",
        default="experiments/B0.5/pool_natural/pool_natural.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/B0.5/pool_natural/networks",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"Extracting {len(df):,} networks to {out_dir} …")

    for _, row in df.iterrows():
        record = {
            "network_id":      int(row["network_id"]),
            "seed":            int(row["seed"]),
            "valid":           bool(row["valid"]),
            "baseline_reward": float(row["baseline_reward"]),
            "baseline_fitness": float(row["baseline_fitness"]),
            "stratum":         row["stratum"],
            "genome":          json.loads(row["genome_json"]),
            "network_stats": {
                "neurons":     int(row["n_neurons"]),
                "connections": int(row["n_connections"]),
            },
            "rollout_data": {
                "rewards":    json.loads(row["rollout_rewards"]),
                "lengths":    json.loads(row["rollout_rewards"]),  # same as rewards for CartPole
                "eval_seeds": json.loads(row["eval_seeds"]),
            },
        }
        path = out_dir / f"network_{row['network_id']:05d}.json"
        path.write_text(json.dumps(record, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
