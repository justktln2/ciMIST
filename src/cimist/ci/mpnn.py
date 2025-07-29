from typing import NamedTuple
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
from functools import partial
from jax import vmap


class ResidueStates(NamedTuple):
    n_states: ArrayLike
    states_traj: Array
    state_counts: Array

def samples_to_states(mpnn_samples):
    states_traj = jnp.array(mpnn_samples["S"].argmax(axis=1))
    n_states = 20*jnp.ones((states_traj.shape[0],1))
    _, counts = vmap(partial(jnp.unique,
                size=20, fill_value=-1, return_counts=True
                ))(states_traj)
    return ResidueStates(n_states, states_traj, counts)
