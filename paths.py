"""Repository-level path helpers.

All paths are resolved relative to the repository root (the directory that
contains this file), so helpers work regardless of the current working
directory or how Python is invoked.

Usage in any script::

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))  # scripts/
    # or two levels up for scripts/subdir/
    from paths import result_dir_from_legacy_log, task_target_dir, ...
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

_LEGACY_RESULT_DIRS = {
    'log-sys_id':                    ('results', 'system-identification', 'soil'),
    'log-sys_id_sand':               ('results', 'system-identification', 'sand'),
    'log-abs0':                      ('results', 'trajectory-optimization', 'soil'),
    'log-abs2':                      ('results', 'skill-parameter-optimization', 'soil'),
    'log-abs2_sand':                 ('results', 'skill-parameter-optimization', 'sand'),
    'log-abs2-rg':                   ('results', 'skill-parameter-optimization-rectified-gradient', 'soil'),
    'log-abs2-rg_sand':              ('results', 'skill-parameter-optimization-rectified-gradient', 'sand'),
    'log-abs2-cmamae':               ('results', 'baselines', 'cma-mae', 'soil'),
    'log-abs2-cmamae_sand':          ('results', 'baselines', 'cma-mae', 'sand'),
    'log-abs2-sac':                  ('results', 'baselines', 'sac', 'soil'),
    'log-grad-analysis':             ('results', 'analysis', 'gradient'),
    'log-substep-grad-analysis':     ('results', 'analysis', 'substep-gradient'),
    'log-loss-analysis':             ('results', 'analysis', 'loss-landscape'),
    'log-abs2-old-demo':             ('results', 'legacy', 'skill-parameter-optimization', 'soil-old-demo'),
    'log-abs2-old_grad':             ('results', 'legacy', 'skill-parameter-optimization', 'soil-old-gradient'),
    'log-abs2_sand-old-demo':        ('results', 'legacy', 'skill-parameter-optimization', 'sand-old-demo'),
    'log-abs2_sand-old_grad':        ('results', 'legacy', 'skill-parameter-optimization', 'sand-old-gradient'),
}


def repo_root() -> str:
    return str(_REPO_ROOT)


def repo_path(*parts) -> str:
    return str(_REPO_ROOT.joinpath(*parts))


def _material_suffix(material) -> str:
    """Normalise material argument to 'soil' or 'sand'."""
    if material in (True, '_sand', 'sand'):
        return 'sand'
    return 'soil'


# ── data ────────────────────────────────────────────────────────────────────

def trajectories_dir() -> str:
    return repo_path('data', 'trajectories')


def task_target_dir(material='soil') -> str:
    return repo_path('data', 'task-targets', _material_suffix(material))


def system_identification_target_dir(material='soil') -> str:
    return repo_path('data', 'system-identification-targets', _material_suffix(material))


def data_config_path(filename) -> str:
    return repo_path('data', 'configs', filename)


def data_calibration_path(filename) -> str:
    return repo_path('data', 'calibration', filename)


# ── results ──────────────────────────────────────────────────────────────────

def result_dir_from_legacy_log(log_name) -> str:
    """Return the canonical results directory for a legacy ``log-*`` name."""
    parts = _LEGACY_RESULT_DIRS.get(log_name)
    if parts is None:
        raise KeyError(f'Unknown legacy result directory: {log_name!r}')
    return repo_path(*parts)


# ── archive ───────────────────────────────────────────────────────────────────

def render_output_dir(*parts) -> str:
    return repo_path('archive', 'render_outputs', *parts)
