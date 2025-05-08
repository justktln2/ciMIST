import os
from multiprocessing import cpu_count
from . import vmm
from . import dbscan

n_cpus = cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_cpus // 2}"

__all__ = [
    "vmm",
    "dbscan",
]
