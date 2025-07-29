from typing import NamedTuple
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp


class ResidueStates(NamedTuple):
    n_states: ArrayLike
    states_traj: Array
    state_counts: Array

def process_mpnn_samples(mpnn_samples):
    states_traj = jnp.array(mpnn_samples["S"].argmax(axis=1))
    n_states = 20*jnp.ones(states_traj.shape[0])
    states, counts = jnp.unique(states_traj,
                size=20, fill_value=-1, return_counts=True
                )
    return ResidueStates(n_states, states_traj, counts)
