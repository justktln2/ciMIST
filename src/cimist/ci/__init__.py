import os

import os, sys
from multiprocessing import cpu_count
from typing import Union

n_cpus = cpu_count()
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_cpus // 2}'

from . import vmm, dbscan, mpnn

ResidueStates = Union[dbscan.ResidueStates, mpnn.ResidueStates]