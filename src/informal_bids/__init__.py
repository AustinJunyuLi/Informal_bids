"""
Informal Bids: MCMC Estimation for Auction Cutoffs

Tools for simulating and estimating informal bid admission cutoffs
in auctions using Markov Chain Monte Carlo methods.

Modules live under src/informal_bids/:
- data.py: Data classes and generators for simulated data
- data_io.py: Data loading for real auction data
- samplers.py: MCMC sampler implementations
- analysis.py: Results analysis and metrics
- visualization.py: Publication-quality plotting
- sensitivity.py: Sensitivity analysis framework
- cli.py: Command-line entry points
- config.py: Configuration and path constants
- utils.py: Statistical utilities
- numba_kernels.py: Performance-optimized MCMC kernels
"""

from .data import (
    TaskADGPParameters,
    TaskBDGPParameters,
    MCMCConfig,
    TaskAAuctionData,
    TaskBAuctionData,
    TaskADataGenerator,
    TaskBDataGenerator,
)

from .data_io import RealDataLoader

from .samplers import TaskAMCMCSampler, TaskBMCMCSampler

from .analysis import TaskAResultsAnalyzer, TaskBResultsAnalyzer

from .visualization import TaskAVisualizer, TaskBVisualizer

from .sensitivity import TaskASensitivityAnalysis, TaskBSensitivityAnalysis

from .config import configure_plotting

__version__ = "0.2.0"

__all__ = [
    # Data classes
    "TaskADGPParameters",
    "TaskBDGPParameters",
    "MCMCConfig",
    "TaskAAuctionData",
    "TaskBAuctionData",
    # Generators
    "TaskADataGenerator",
    "TaskBDataGenerator",
    # I/O
    "RealDataLoader",
    # Samplers
    "TaskAMCMCSampler",
    "TaskBMCMCSampler",
    # Analysis
    "TaskAResultsAnalyzer",
    "TaskBResultsAnalyzer",
    # Visualization
    "TaskAVisualizer",
    "TaskBVisualizer",
    # Sensitivity
    "TaskASensitivityAnalysis",
    "TaskBSensitivityAnalysis",
    # Config
    "configure_plotting",
]
