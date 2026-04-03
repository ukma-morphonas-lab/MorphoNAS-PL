#!/usr/bin/env python3
"""
Consolidate B0.5+ data: merge original grid subset + extended sweep into one dataset.

Steps:
  1. Convert extended sweep JSONs to parquet (per grid point)
  2. Copy them into the main B0.5+ sweep directory alongside original-grid parquets
  3. Verify no overlap between original and extended grid points

Usage:
  python scripts/consolidate_B0_5plus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


B05P_ROOT = Path("experiments/B0.5+")
B05P_SWEEP = B05P_ROOT / "sweep"              # original 75 grid points (parquet)
B05P_EXTENDED_DIRS = [
    B05P_ROOT / "sweep_extended",    # v1: 50 grid points (eta extension)
    B05P_ROOT / "sweep_extended_v2", # v2: 107 grid points (decay + eta extension)
]


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
    """Convert all network JSONs in one config dir to a single Parquet file."""
    json_files = sorted(config_dir.glob("network_*.json"))
    if not json_files:
        return 0

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        rows.append(flatten_network_json(data))

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="zstd")
    return len(rows)


def main() -> None:
    available_dirs = [d for d in B05P_EXTENDED_DIRS if d.exists()]
    if not available_dirs:
        print(
            f"ERROR: No extended sweep dirs found. Looked for:\n"
            + "\n".join(f"  {d}" for d in B05P_EXTENDED_DIRS),
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("B0.5+ CONSOLIDATION")
    print("=" * 60)

    # Find existing parquet files (original grid)
    existing_parquets = set(f.name for f in B05P_SWEEP.glob("eta_*.parquet"))
    print(f"\nOriginal grid parquets: {len(existing_parquets)}")

    total_records = 0
    converted = 0

    for ext_dir in available_dirs:
        config_dirs = sorted(d for d in ext_dir.iterdir() if d.is_dir())
        print(f"\n--- {ext_dir.name}: {len(config_dirs)} directories ---")

        for config_dir in config_dirs:
            out_name = f"{config_dir.name}.parquet"
            if out_name in existing_parquets:
                print(f"  SKIP (exists): {out_name}")
                continue

            out_path = B05P_SWEEP / out_name
            n = convert_config_dir(config_dir, out_path)
            if n > 0:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  {config_dir.name}: {n} records -> {size_mb:.2f} MB")
                total_records += n
                converted += 1
                existing_parquets.add(out_name)

    # Summary
    all_parquets = sorted(B05P_SWEEP.glob("eta_*.parquet"))
    total_size_mb = sum(f.stat().st_size for f in all_parquets) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"DONE.")
    print(f"  Converted: {converted} new grid points ({total_records:,} records)")
    print(f"  Total parquets: {len(all_parquets)} ({total_size_mb:.1f} MB)")
    print(f"  Output: {B05P_SWEEP}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
