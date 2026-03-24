#!/usr/bin/env python3
"""
B0.5+ Setup: Create stratified subsample and extract existing data.

Creates:
  1. Subsample pool directory with symlinks to selected networks
  2. Filtered parquet files for the original 75 grid points (subsample only)
  3. Metadata documenting the subsample composition

Usage:
  python scripts/setup_B0_5plus.py [--n-weak 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("code"))

# ── Paths ──────────────────────────────────────────────────────────────
B05_POOL = Path("experiments/B0.5/pool_natural")
B05_SWEEP = Path("experiments/B0.5/sweep")
B05P_ROOT = Path("experiments/B0.5+")
B05P_POOL = B05P_ROOT / "pool_subsample"
B05P_SWEEP = B05P_ROOT / "sweep"


def _load_network_strata(pool_dir: Path) -> dict[int, str]:
    """Return {network_id: stratum} for all networks in pool."""
    networks_dir = pool_dir / "networks"
    mapping: dict[int, str] = {}
    for fpath in sorted(networks_dir.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        nid = int(data.get("network_id", 0))
        stratum = data.get("stratum", "weak")
        mapping[nid] = stratum
    return mapping


def _select_subsample(
    strata_map: dict[int, str], n_weak: int, seed: int
) -> list[int]:
    """Select all non-Weak + n_weak random Weak network IDs."""
    rng = np.random.default_rng(seed)

    non_weak = [nid for nid, s in strata_map.items() if s != "weak"]
    weak_all = np.array([nid for nid, s in strata_map.items() if s == "weak"])

    n_weak_actual = min(n_weak, len(weak_all))
    weak_sample = rng.choice(weak_all, size=n_weak_actual, replace=False).tolist()

    selected = sorted(int(x) for x in non_weak + weak_sample)
    return selected


def _create_symlink_pool(
    selected_ids: set[int], src_pool: Path, dst_pool: Path
) -> int:
    """Create pool directory with symlinks to selected networks."""
    dst_networks = dst_pool / "networks"
    dst_networks.mkdir(parents=True, exist_ok=True)

    count = 0
    for fpath in sorted((src_pool / "networks").glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)
        nid = int(data.get("network_id", 0))
        if nid in selected_ids:
            link = dst_networks / fpath.name
            if not link.exists():
                link.symlink_to(fpath.resolve())
            count += 1
    return count


def _filter_parquet_data(
    sweep_dir: Path, selected_ids: set[int], out_dir: Path
) -> int:
    """Filter sweep parquet files to subsample and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    shard_files = sorted(sweep_dir.glob("eta_*.parquet"))
    parquet_files = shard_files if shard_files else sorted(sweep_dir.glob("*.parquet"))

    for pfile in parquet_files:
        df = pd.read_parquet(pfile)
        df_sub = df[df["network_id"].isin(selected_ids)]
        if len(df_sub) > 0:
            df_sub.to_parquet(out_dir / pfile.name, compression="zstd", index=False)
            total_rows += len(df_sub)
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="B0.5+ subsample setup")
    parser.add_argument("--n-weak", type=int, default=500,
                        help="Number of Weak networks to include (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Weak sampling (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("B0.5+ SUBSAMPLE SETUP")
    print("=" * 60)

    # Step 1: Load strata mapping
    print("\n[1/4] Loading network strata from pool...")
    strata_map = _load_network_strata(B05_POOL)
    strata_counts: dict[str, int] = {}
    for s in strata_map.values():
        strata_counts[s] = strata_counts.get(s, 0) + 1
    print(f"  Full pool: {len(strata_map):,} networks")
    for s, c in sorted(strata_counts.items()):
        print(f"    {s}: {c:,}")

    # Step 2: Select subsample
    print(f"\n[2/4] Selecting subsample (all non-Weak + {args.n_weak} Weak)...")
    selected = _select_subsample(strata_map, args.n_weak, args.seed)
    selected_set = set(selected)

    sub_counts: dict[str, int] = {}
    for nid in selected:
        s = strata_map[nid]
        sub_counts[s] = sub_counts.get(s, 0) + 1
    print(f"  Subsample: {len(selected):,} networks")
    for s, c in sorted(sub_counts.items()):
        print(f"    {s}: {c:,}")

    # Step 3: Create symlink pool
    print(f"\n[3/4] Creating symlink pool at {B05P_POOL}...")
    n_links = _create_symlink_pool(selected_set, B05_POOL, B05P_POOL)
    print(f"  Created {n_links} symlinks")

    # Step 4: Filter existing parquet data
    print(f"\n[4/4] Filtering existing sweep data to subsample...")
    n_rows = _filter_parquet_data(B05_SWEEP, selected_set, B05P_SWEEP)
    print(f"  Extracted {n_rows:,} rows across 75 grid points")

    # Write metadata
    metadata = {
        "experiment": "B0.5+",
        "description": "Stratified subsample from B0.5 + extended eta/decay grid",
        "parent": "B0.5",
        "subsample_seed": args.seed,
        "n_weak_target": args.n_weak,
        "subsample_composition": sub_counts,
        "total_networks": len(selected),
        "original_grid_rows": n_rows,
        "network_ids": selected,
    }
    B05P_ROOT.mkdir(parents=True, exist_ok=True)
    with open(B05P_ROOT / "subsample_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also write a network ID list for easy consumption
    with open(B05P_ROOT / "network_ids.txt", "w") as f:
        for nid in selected:
            f.write(f"{nid}\n")

    print(f"\n{'=' * 60}")
    print(f"DONE. Subsample ready at {B05P_ROOT}")
    print(f"  Pool: {B05P_POOL}")
    print(f"  Sweep (original grid): {B05P_SWEEP}")
    print(f"  Network list: {B05P_ROOT / 'network_ids.txt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
