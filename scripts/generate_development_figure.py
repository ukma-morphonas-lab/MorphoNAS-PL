#!/usr/bin/env python3
"""Generate a 3-panel developmental time-series figure for the paper.

Shows morphogen fields + cells + connections at three developmental stages
(early, mid, final) for a single genome on a 10×10 grid.

Outputs a single-column-width PDF suitable for ACM sigconf (3.33 inches wide).

Usage:
    .venv/bin/python scripts/generate_development_figure.py
    .venv/bin/python scripts/generate_development_figure.py --genome path/to/genome.json
    .venv/bin/python scripts/generate_development_figure.py --steps 2,10,200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType (no Type-3)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 9
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.abspath("code"))

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid

# ACM sigconf single-column width
COL_WIDTH_IN = 3.33

# Default genome: K01_cartpole_R11 from the original MorphoNAS Experiment B
# 10×10 grid, 3 morphogens, 200 growth steps — matches the paper's setup
DEFAULT_GENOME = None  # Will search common locations

# Default developmental snapshots
DEFAULT_STEPS = [2, 10, 200]


def find_default_genome() -> Path:
    """Search common locations for the R11 genome."""
    candidates = [
        Path("experiments/MoprhoNAS_ExpB/original/K01_cartpole_R11/best_genome.json"),
        Path("../MorphoNAS-PL_private/experiments/MoprhoNAS_ExpB/original/K01_cartpole_R11/best_genome.json"),
        Path("../MorphoNAS/experiments/_ExpB_RNN_controller/results/K01_cartpole_R11/best_genome.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Default genome not found. Provide --genome path/to/genome.json"
    )


def draw_panel(ax, grid: Grid, step: int, label: str):
    """Draw one developmental snapshot panel."""
    # Build RGB from morphogen concentrations (up to 3 channels)
    rgb = np.ones((grid.size_x, grid.size_y, 3))
    for i in range(min(3, grid.num_morphogens)):
        rgb[:, :, i] -= np.clip(grid.get_morphogen_array(i), 0, 1)

    ax.imshow(rgb, vmin=0, vmax=1, interpolation="nearest")

    # Overlay cells
    neuron_ids = grid.get_neuron_ids()
    for cell_id in grid.get_cell_ids():
        x, y = grid.get_cell_position(cell_id)
        is_progenitor = grid.is_progenitor(cell_id)
        rect = patches.Rectangle(
            (y - 0.45, x - 0.45), 0.9, 0.9,
            linewidth=0.8,
            edgecolor="white",
            facecolor="none",
            linestyle=":" if is_progenitor else "-",
        )
        ax.add_patch(rect)

    # Overlay connections
    src_ids, tgt_ids = grid.neuron_connections.nonzero()
    for src_id, tgt_id in zip(src_ids + 1, tgt_ids + 1):
        if src_id in neuron_ids and tgt_id in neuron_ids:
            src_pos = grid.get_cell_position(src_id)
            tgt_pos = grid.get_cell_position(tgt_id)
            ax.plot(
                [src_pos[1], tgt_pos[1]],
                [src_pos[0], tgt_pos[0]],
                color="white", alpha=0.5, linewidth=0.4,
            )

    ax.set_xlim(-0.5, grid.size_y - 0.5)
    ax.set_ylim(-0.5, grid.size_x - 0.5)  # (0,0) at bottom-left, neurons near top
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Labels
    n_neurons = grid.neuron_count()
    n_conn = grid.neuron_connections.nnz
    ax.set_title(label, fontsize=9, pad=3)
    ax.text(
        0.03, 0.03, f"{n_neurons}N, {n_conn}C",
        transform=ax.transAxes, fontsize=8, color="white",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate developmental time-series figure"
    )
    parser.add_argument(
        "--genome", type=str, default=None,
        help="Path to genome JSON file (default: K01_cartpole_R11)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated step numbers to capture (default: 2,10,200)",
    )
    parser.add_argument(
        "--output", type=str,
        default="../PhD-thesis/papers/gecco2026-evoself/figures/morphonas_development.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    # Load genome
    if args.genome:
        genome_path = Path(args.genome)
    else:
        genome_path = find_default_genome()
    print(f"Genome: {genome_path}")

    with open(genome_path) as f:
        genome = Genome.from_dict(json.load(f))
    print(f"Grid: {genome.size_x}×{genome.size_y}, steps: {genome.max_growth_steps}, morphogens: {genome.num_morphogens}")

    # Parse steps
    steps = DEFAULT_STEPS if args.steps is None else [int(s) for s in args.steps.split(",")]
    n_panels = len(steps)
    print(f"Snapshots at steps: {steps}")

    # Create figure: 3 panels side by side, single column width
    panel_w = COL_WIDTH_IN / n_panels
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(COL_WIDTH_IN, panel_w + 0.25),  # slightly taller for title
        dpi=300,
    )
    if n_panels == 1:
        axes = [axes]

    # Grow the network step by step, capturing at requested steps
    grid = Grid(genome)
    grid.add_cell((grid.size_x // 2, grid.size_y // 2), "progenitor")

    step_idx = 0
    captured = 0

    # Capture step 0 if requested
    if steps[0] == 0:
        draw_panel(axes[captured], grid, 0, f"Step 0")
        captured += 1

    for s in range(1, genome.max_growth_steps + 1):
        grid.step()

        if captured < n_panels and s == steps[captured]:
            label = f"Step {s}"
            if s == genome.max_growth_steps:
                label = f"Step {s} (final)"
            draw_panel(axes[captured], grid, s, label)
            captured += 1
            print(f"  Step {s}: {grid.neuron_count()} neurons, {grid.neuron_connections.nnz} connections")

    # Final step (self-connections etc.)
    if captured < n_panels and steps[captured] >= genome.max_growth_steps:
        grid.final_step()
        draw_panel(axes[captured], grid, genome.max_growth_steps, f"Step {genome.max_growth_steps} (final)")
        captured += 1
        print(f"  Final: {grid.neuron_count()} neurons, {grid.neuron_connections.nnz} connections")

    plt.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.88, bottom=0.01)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
