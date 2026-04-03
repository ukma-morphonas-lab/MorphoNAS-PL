#!/usr/bin/env python3
"""
One-time migration: convert per-network JSON sweep results into Parquet files.

Input:  experiments/B0.5/sweep/<config>/network_*.json
Output: experiments/B0.5/sweep_parquet/<config>.parquet
        + sweep_summary.json and sweep.log copied as-is

After validation, the original sweep/ directory can be removed and
sweep_parquet/ renamed to sweep/.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SWEEP_DIR = Path("experiments/B0.5/sweep")
OUT_DIR = Path("experiments/B0.5/sweep_parquet")


def flatten_network_json(data: dict) -> dict:
    """Flatten nested JSON into a single-level dict for tabular storage."""
    row = {
        "network_id": data["network_id"],
        "stratum": data["stratum"],
        "original_seed": data["original_seed"],
        "baseline_reward_stored": data["baseline_reward_stored"],
        "eta": data["eta"],
        "decay": data["decay"],
        "rollouts": data["rollouts"],
        "reset_plastic_each_episode": data["reset_plastic_each_episode"],
        "num_neurons": data["network_stats"]["num_neurons"],
        "num_connections": data["network_stats"]["num_connections"],
        "timestamp": data["timestamp"],
        # baseline
        "baseline_rewards": data["baseline"]["rewards"],
        "baseline_lengths": data["baseline"]["lengths"],
        "baseline_avg_reward": data["baseline"]["avg_reward"],
        "baseline_std_reward": data["baseline"]["std_reward"],
        "baseline_min_reward": data["baseline"]["min_reward"],
        "baseline_max_reward": data["baseline"]["max_reward"],
        "baseline_fitness": data["baseline"]["fitness"],
        # plastic
        "plastic_rewards": data["plastic"]["rewards"],
        "plastic_lengths": data["plastic"]["lengths"],
        "plastic_avg_reward": data["plastic"]["avg_reward"],
        "plastic_std_reward": data["plastic"]["std_reward"],
        "plastic_min_reward": data["plastic"]["min_reward"],
        "plastic_max_reward": data["plastic"]["max_reward"],
        "plastic_fitness": data["plastic"]["fitness"],
        "mean_abs_delta_per_step": data["plastic"]["plasticity"]["mean_abs_delta_per_step"],
        "max_abs_delta_per_step": data["plastic"]["plasticity"]["max_abs_delta_per_step"],
        # summary
        "delta_fitness": data["summary"]["delta_fitness"],
        "delta_reward": data["summary"]["delta_reward"],
        "improved": data["summary"]["improved"],
    }
    return row


def convert_config_dir(config_dir: Path, out_path: Path) -> int:
    """Convert all network JSONs in one config dir to a single Parquet file.

    Returns the number of records written.
    """
    json_files = sorted(config_dir.glob("network_*.json"))
    if not json_files:
        return 0

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        rows.append(flatten_network_json(data))

    df = pd.DataFrame(rows)
    # Store list columns as PyArrow list types for compact representation
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")
    return len(rows)


def main():
    if not SWEEP_DIR.exists():
        print(f"ERROR: {SWEEP_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy non-directory files (sweep_summary.json, sweep.log)
    for f in SWEEP_DIR.iterdir():
        if f.is_file():
            dest = OUT_DIR / f.name
            dest.write_bytes(f.read_bytes())
            print(f"  copied {f.name}")

    # Convert each config directory
    config_dirs = sorted(d for d in SWEEP_DIR.iterdir() if d.is_dir())
    total_records = 0
    total_configs = 0

    for i, config_dir in enumerate(config_dirs, 1):
        out_path = OUT_DIR / f"{config_dir.name}.parquet"
        n = convert_config_dir(config_dir, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        total_records += n
        total_configs += 1
        print(f"  [{i}/{len(config_dirs)}] {config_dir.name}: {n} records -> {size_mb:.2f} MB")

    # Summary
    total_size_mb = sum(
        f.stat().st_size for f in OUT_DIR.glob("*.parquet")
    ) / (1024 * 1024)

    print(f"\nDone: {total_configs} configs, {total_records} records")
    print(f"Total Parquet size: {total_size_mb:.1f} MB")
    print(f"Original JSON size: ~14 GB")
    print(f"Compression ratio: ~{14000 / total_size_mb:.0f}x")


if __name__ == "__main__":
    main()
