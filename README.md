# DDBot

**Differentiable Physics-based Digging Robot for Unknown Granular Materials**

[![Paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/document/11270989)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=eoNx5V688H0)

> Xintong Yang, Minglun Wei, Yu-Kun Lai, Ze Ji  
> *IEEE Robotics and Automation Letters*, 2025

---

<!-- System diagram placeholder — replace the line below with your diagram image -->
![System Diagram](docs/system_diagram.png)

---

DDBot is a differentiable physics-based framework for high-precision manipulation of unknown granular materials (soil and sand). It combines a differentiable granular simulator (DOMA) with gradient-based system identification and skill/trajectory optimisation to enable a robot to precisely dig and place granular material without prior knowledge of its physical properties.

This repository contains the simulator, experiment scripts, datasets, figures, and result artefacts for the experiments reported in the paper.

## Repository layout

```
.
├── simulator/          # doma Python package — differentiable granular simulator
│   └── doma/
├── scripts/            # experiment runners, plotting, analysis utilities
├── data/
│   ├── calibration/        # camera extrinsics
│   ├── configs/            # CMA-MAE and RL agent config files, skill definitions
│   ├── system-identification-targets/   # target point clouds for sys-id (soil & sand)
│   ├── task-targets/       # target point clouds for manipulation tasks (soil & sand)
│   └── trajectories/       # MoveIt trajectory files
├── results/            # all experiment outputs
│   ├── system-identification/
│   ├── skill-parameter-optimization/
│   ├── skill-parameter-optimization-rectified-gradient/
│   ├── trajectory-optimization/
│   ├── baselines/          # cma-mae, sac
│   └── analysis/           # gradient, substep-gradient, loss-landscape
├── figs/               # canonical figures (paper and supplementary)
├── archive/            # render outputs and legacy artefacts
├── paths.py            # centralised path helpers for all scripts
├── pyproject.toml
├── requirements.txt
└── environment.yml
```

## Quickstart

### 1. Install Git LFS

This repository tracks large binary assets (point clouds, model weights, videos) with Git LFS. Install it before cloning:

```bash
git lfs install
git clone https://github.com/IanYangChina/DDBot
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate DPS
```

### 3. Install the package

```bash
pip install -e .
```

### 4. (RL baselines only) Install DRL_Implementation

The SAC/RL workflows depend on a companion repository:

```bash
git clone https://github.com/IanYangChina/DRL_Implementation
# Add to PYTHONPATH or install into the same environment
export PYTHONPATH="$PYTHONPATH:/path/to/DRL_Implementation"
```

## Reproducing experiments

### System identification

```bash
python scripts/run_si.py
```

Key settings used in the paper: particle density `5e6`, height-map loss, gradient clipping, line search, manual initialisation, resolution `40`.

Outputs → `results/system-identification/{soil,sand}/`

### Skill-parameter optimisation

```bash
python scripts/run_so_abs2.py
```

Outputs → `results/skill-parameter-optimization/{soil,sand}/`  
Rectified-gradient variant → `results/skill-parameter-optimization-rectified-gradient/{soil,sand}/`

### Trajectory-level optimisation

```bash
python scripts/run_so_abs0.py
```

Outputs → `results/trajectory-optimization/soil/`

### Baselines

```bash
# CMA-MAE
python scripts/run_cmamae.py        # → results/baselines/cma-mae/{soil,sand}/

# SAC (requires DRL_Implementation)
python scripts/run_rl_abs2.py       # → results/baselines/sac/soil/
```

### Gradient and loss analysis

```bash
# Gradient assessment scripts
python scripts/gradient_loss_assessment/...

# Read TensorBoard events / aggregate logs
python scripts/data_generation/read_tb_event.py
```

Outputs → `results/analysis/{gradient,substep-gradient,loss-landscape}/`

## Figures

`figs/` is the canonical figure directory. Quantitative figures can be traced back to the logs in `results/` plus the post-processing scripts in `scripts/gradient_loss_assessment/` and `scripts/data_generation/`.

## Citation

If you use this work, please cite:

```bibtex
@article{yang2025ddbot,
  title   = {DDBot: Differentiable Physics-based Digging Robot for Unknown Granular Materials},
  author  = {Yang, Xintong and Wei, Minglun and Lai, Yu-Kun and Ji, Ze},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2025},
  doi     = {10.1109/LRA.2025.XXXXXXX},
  url     = {https://ieeexplore.ieee.org/document/11270989}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).
