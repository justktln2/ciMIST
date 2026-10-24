"""Opt-in lossless pair storage; this is not an array-compatible wrapper."""
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class SparseProbability:
    shape: tuple
    dtype: str
    positions: np.ndarray
    values: np.ndarray

    @classmethod
    def encode(cls, probability):
        array = np.asarray(probability)
        if array.ndim != 2 or array.dtype.kind != "f":
            raise ValueError("Sparse probabilities require a floating matrix")
        if not np.isfinite(array).all() or (array < 0).any() or np.signbit(array).any():
            raise ValueError("Sparse probabilities require finite nonnegative values without negative zero")
        index_dtype = (np.uint16 if array.size <= 2**16 else
                       np.uint32 if array.size <= 2**32 else np.uint64)
        positions = np.flatnonzero(array).astype(index_dtype)
        values = array.ravel()[positions].copy()
        positions.flags.writeable = values.flags.writeable = False
        return cls(array.shape, array.dtype.str, positions, values)

    def dense_numpy(self):
        array = np.zeros(self.shape, dtype=np.dtype(self.dtype))
        array.ravel()[self.positions] = self.values
        return array


def dense_probability(value):
    """Return original padded JAX values before scientific arithmetic."""
    if not isinstance(value, SparseProbability):
        return value
    result = jnp.asarray(value.dense_numpy())
    if np.dtype(result.dtype) != np.dtype(value.dtype):
        raise ValueError("JAX precision configuration cannot preserve probability dtype")
    return result
