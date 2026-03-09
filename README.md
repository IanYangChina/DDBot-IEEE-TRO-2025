# DDBot

Differentiable Physics-based Digging Robot for unknown granular materials.

This repository contains the simulator, experiment scripts, datasets, figures, and result artefacts used for the DDBot project on high-precision soil and sand manipulation. The codebase supports differentiable system identification, skill-parameter optimization, trajectory-level optimization, and baseline methods for the experiments reported in the paper.

The repository is currently being reorganized into clearer public-release directories. Public result families are being grouped under `results/`, while old `log-*` paths may remain temporarily as compatibility links during the transition.

## What is in this repository

- **`simulator/`**: the current Python package source for `doma`, the differentiable granular manipulation simulator.
- **`scripts/`**: experiment runners, plotting scripts, analysis utilities, and manual validation scripts.
- **`data/`**: target point clouds, calibration files, trajectories, and experiment configuration files.
- **`figs/`**: the canonical figure directory for paper and project figures.
- **`results/`**: clearer public result layout for system identification, optimization, baselines, and analysis artefacts.
- **`log-*`**: legacy compatibility paths during the transition to the clearer `results/` layout.
- **`archive/render_outputs/`**: archived generated render outputs.

## Naming conventions used in the results

The existing result directories use short experiment names. For clarity:

- **`abs0`** = trajectory-level optimization
- **`abs2`** = skill-parameter optimization
- **`log-abs2`** = soil results by default
- **`*_sand`** = sand variants of the corresponding experiment family
- **`log-sys_id*`** = system-identification experiments
- **`log-grad-analysis*` / `log-loss-analysis*`** = gradient and loss-analysis artefacts

These names will be made clearer during the repository restructure, but the current scripts and results still use them.

## Status

This repository is currently being cleaned for open-source release. The main code and result artefacts are public, but some cleanup is still in progress:

- large binary assets are being prepared for Git LFS
- result directories are being renamed for clarity
- root-level packaging is being added
- temporary planning/reference material will be removed before the final public release shape

## Quickstart

### 1. Install Git LFS

This repository contains large public assets. Install Git LFS before cloning or before pulling large files:

```bash
git lfs install
```

### 2. Create the environment

The current repository validation uses the existing `DPS` Conda environment with Python 3.7:

```bash
conda env create -f environment.yml
conda activate DPS
```

### 3. Install the package from the repository root

```bash
pip install -e .
```

### 4. Install the companion DRL repository for RL workflows

Some RL and plotting workflows depend on the companion repository:

- `https://github.com/IanYangChina/DRL_Implementation`

Until that dependency is integrated more cleanly, clone it separately and make it importable in your environment, for example by placing it alongside this repository and exporting `PYTHONPATH`, or by installing it into the same environment if supported by that repository.

## Core workflows

### System identification

Main script:

```bash
python scripts/run_si.py
```

Typical reported settings use:

- particle density `5e6`
- height-map loss
- gradient clipping
- line search
- manual initialization
- resolution `40`

Outputs are organized under:

- `results/system-identification/soil/`
- `results/system-identification/sand/`

### Skill-parameter optimization

Main script:

```bash
python scripts/run_so_abs2.py
```

Outputs are organized under:

- `results/skill-parameter-optimization/soil/`
- `results/skill-parameter-optimization/sand/`
- `results/skill-parameter-optimization-rectified-gradient/soil/`
- `results/skill-parameter-optimization-rectified-gradient/sand/`

### Trajectory-level optimization

Main script:

```bash
python scripts/run_so_abs0.py
```

Outputs are organized under:

- `results/trajectory-optimization/soil/`

### Baselines

#### CMA-MAE

```bash
python scripts/run_cmamae.py
```

Outputs are organized under:

- `results/baselines/cma-mae/soil/`
- `results/baselines/cma-mae/sand/`

#### SAC / RL

```bash
python scripts/run_rl_abs2.py
```

Outputs are organized under:

- `results/baselines/sac/soil/`

This workflow requires the external `DRL_Implementation` repository described above.

### Gradient and loss analysis

Scripts are under:

- `scripts/gradient_loss_assessment/`
- `scripts/data_generation/read_tb_event.py`

Analysis outputs are organized under:

- `results/analysis/gradient/`
- `results/analysis/substep-gradient/`
- `results/analysis/loss-landscape/`

## Figures and reproducibility

- **`figs/` is the canonical figure directory.**
- Quantitative figures can mostly be traced back to existing logs plus post-processing scripts.
- Some paper-ready montage assets appear to have manual assembly steps and are being documented as part of the release cleanup.

## Packaging notes

The repository currently exposes the `doma` Python package from `simulator/doma/`. A root `pyproject.toml` is provided so the package can be installed from the repository root during the cleanup process, and the current validation target is the `DPS` environment. The package layout will be normalized further in the next restructure phase.

## Known gaps

The current repository still contains a few issues that affect full from-scratch reproduction:

- some scripts import `doma.envs.planting_env`, while the repository currently provides `planting_env_v1.py`
- `simulator/doma/envs/__init__.py` still references stale modules
- RL workflows require the external `DRL_Implementation` repository

These are being addressed during the public-release cleanup.

## Citation

If you use this repository, please cite the DDBot paper. A machine-readable citation file is provided in `CITATION.cff`.
