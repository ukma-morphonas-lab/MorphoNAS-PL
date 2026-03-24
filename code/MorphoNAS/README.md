# MorphoNAS: Embryogenic Neural Architecture Search Through Morphogen-Guided Development

[![arXiv](https://img.shields.io/badge/arXiv-2507.13785-b31b1b.svg)](https://arxiv.org/abs/2507.13785)

This repository contains the reference implementation and experiment scripts for the paper:

MorphoNAS: Embryogenic Neural Architecture Search Through Morphogen-Guided Development, submitted for peer review. Preprint: [arXiv:2507.13785v1](https://arxiv.org/abs/2507.13785v1).

Authors: Mykola Glybovets, Sergii Medvid

## Overview

MorphoNAS is a morphogenetic neural architecture search (NAS) system. Compact genomes encode morphogen dynamics and threshold-based developmental rules that deterministically grow a neural network from a single progenitor cell via local chemical interactions. We evaluate MorphoNAS in two domains:

- Structural targeting: evolving genomes that develop into predefined random graph configurations (8-31 nodes)
- Functional performance: solving CartPole with compact evolved RNN controllers (6-7 neurons under size pressure)

The codebase includes experiment pipelines, analysis scripts, and utilities to reproduce the results reported in the preprint.

## Repository structure

- `src/`
  - `experimentA.py`: structural targeting experiments (graph properties)
  - `experimentB.py`: RNN controller experiments (CartPole)
  - `regenerate_displays.py`: regenerate visualizations for experiment outputs
  - Additional modules: genome, evolution, morphogen and neural propagation, displays, etc.
- `experiments/`
  - `_ExpA_graph_properties/`: configs and results for Experiment A
  - `_ExpB_RNN_controller/`: configs and results for Experiment B
- `experimentA.sh`, `experimentB.sh`: end-to-end scripts for setup, running, and post-processing
- `requirements.txt`: Python dependencies

## Quick start

1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Run experiments via helper scripts

```bash
# Experiment A: structural targeting
bash experimentA.sh

# Experiment B: RNN controller (CartPole)
bash experimentB.sh
```

Each script will set up the environment, run the experiments, and regenerate displays.

### Running components manually

You can also run components directly:

```bash
# Experiment A (generate configs, run, then regenerate displays)
python src/experimentA.py --generate-configs --run-experiments --seed=54210 --experiment-seed=65420
python src/regenerate_displays.py experiments/_ExpA_graph_properties/results/*/

# Experiment B (run multiple trials, regenerate displays, and run analyses)
python src/experimentB.py --run-experiments --experiment-seed=65420 --num-runs=100
python src/regenerate_displays.py experiments/_ExpB_RNN_controller/results/*/
python src/experimentB_analysis.py experiments/_ExpB_RNN_controller K01_cartpole
python src/experimentB_analysis.py experiments/_ExpB_RNN_controller K02_cartpole_min
```

Outputs (plots, metrics, and intermediate JSON/CSV artifacts) are stored under the corresponding `experiments/*/results/` directories.

### Reproducing figures and analysis

- Experiment A visualizations are regenerated via `src/regenerate_displays.py` over `experiments/_ExpA_graph_properties/results/*/`.
- Experiment B includes additional analyses:
  - `src/experimentB_analysis.py experiments/_ExpB_RNN_controller K01_cartpole`
  - `src/experimentB_analysis.py experiments/_ExpB_RNN_controller K02_cartpole_min`

### Citation

If you use this repository or build upon MorphoNAS, please cite the preprint:

```bibtex
@misc{glybovets2025morphonas,
  title        = {MorphoNAS: Embryogenic Neural Architecture Search Through Morphogen-Guided Development},
  author       = {Mykola Glybovets and Sergii Medvid},
  year         = {2025},
  eprint       = {2507.13785},
  archivePrefix= {arXiv},
  primaryClass = {cs.NE},
  url          = {https://arxiv.org/abs/2507.13785}
}
```

### Notes

- Python version: it was run on Python 3.13. See `requirements.txt` for dependencies.
- The experiments were run on a single machine in CPU-only mode. CPU: Apple M2 Pro, RAM: 96GB.
- Random seeds can be controlled via the CLI flags shown above to support reproducibility.

### Contact and issues

For questions, bug reports, or feature requests, please open an issue in this repository. For paper-related inquiries, refer to the contact information on the preprint page: [arXiv:2507.13785v1](https://arxiv.org/abs/2507.13785v1).
