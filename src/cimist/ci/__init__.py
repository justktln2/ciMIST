"""
The ci (conformational inference) submodule of ciMIST provides tools for
estimating the conformational ensemble of biomolecules from molecular dynamics
simulations using the MIST algorithm. This module includes vector map
implementations of the Von-Mises Mixture Model (VMM) and Density-Based Spatial
Clustering of Applications with Noise (DBSCAN) algorithms, which are used to
infer  the conformational states of biomolecules from simulation data.
For more information, see the documentation at:
https://github.com/justktln2/cimist
"""

import os
import sys
from multiprocessing import cpu_count
from . import vmm, dbscan

n_cpus = cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpus // 2}"

__all__ = ["vmm", "dbscan"]
