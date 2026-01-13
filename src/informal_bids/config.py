"""
Global configuration for plotting, paths, and constants.

This module centralizes all configuration that was previously duplicated
across task_a_mcmc.py, task_b_mcmc.py, and sensitivity analysis files.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress warnings globally
warnings.filterwarnings('ignore')


def configure_plotting():
    """Set up publication-quality plotting defaults.

    Call this function at the start of any script that generates plots.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,
        'mathtext.fontset': 'cm',
        'mathtext.fallback': 'cm',
    })


# Path constants
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
SENSITIVITY_DIR = OUTPUTS_DIR / "sensitivity"

# Task-specific output directories
TASK_A_BASELINE_DIR = BASELINE_DIR / "task_a"
TASK_B_BASELINE_DIR = BASELINE_DIR / "task_b"
TASK_A_SENSITIVITY_DIR = SENSITIVITY_DIR / "task_a"
TASK_B_SENSITIVITY_DIR = SENSITIVITY_DIR / "task_b"
