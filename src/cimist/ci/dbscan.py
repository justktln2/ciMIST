"""
DBSCAN operations for inferring conformations from von Mises mixture models
using density overlap.
"""

import jax.numpy as jnp
import jax
from jax.scipy.special import entr
from typing import NamedTuple, Optional
from jax import Array, jit, vmap
from jax.typing import ArrayLike
import cimist.entropy as ee
from cimist.utils.chunked_vmap import vmap_chunked as cvmap
from cimist.ci.vmm import MixtureFitState


class ResidueStates(NamedTuple):
    """
    Represents residue conformations, including trajectories of conformations,
    which are stored in the attribute states_traj.

    Attributes
    ----------
    n_states: ArrayLike, the number of states inferred by DBSAN
    S_mle: ArrayLike, the "plug-in" estimator of the entropy estimated from the maximum likelihood states.
    S_se: ArrayLike, the estimated standard error of the entropy estimate
    states_traj: Array, the trajectory of states
    state_counts: Array, the number of times each state is observed the trajetory
    dbscan_S: ArrayLike, the entropy estimated from the summed mixture component weights
    dbscan_p: ArrayLike, the summed mixture component weights for each state
    dbscan_eps: ArrayLike, the eps parameter used for DBSCAN
    dbscan_min_probability_mass, float, the minimum probability mass used in DBSCAN
    C_dbscan: Optional[Array]=None, the coarse-graining operator which was applied
              to the von Mises mixture components to produce these states.
    """

    n_states: ArrayLike
    S_mle: ArrayLike
    S_se: ArrayLike
    states_traj: Array
    state_counts: Array
    dbscan_S: ArrayLike
    dbscan_p: ArrayLike
    dbscan_eps: ArrayLike
    dbscan_min_probability_mass: float
    C_dbscan: Optional[Array] = None


class EntropyTrace(NamedTuple):
    S_mle: ArrayLike
    S_vn: ArrayLike
    S_pos_mean: ArrayLike
    S_pos_se: ArrayLike
    Z_sq_vn: ArrayLike
    Eps: ArrayLike


class CoarseGraining(NamedTuple):
    """
    Represents a DBSCAN-based coarse-graining of von Mises mixture components into conformations.

    Attributes
    ----------
    C: Array, the coarse-graining operator which applied
            to the von Mises mixture components.
    C_core: Array, the coarse-graining operator which applied
            to the core von Mises mixture components.
    C_non_noise: Array, the coarse-graining operator which applied
            to the non_noise von Mises mixture components.
    w: Array, the coarse-graining operator which applied
            to the non_noise von Mises mixture components.
    S: ArrayLike, the "plug-in" estimator of the entropy estimated from the maximum likelihood states.
    S_vn: ArrayLike, the von Neumann entropy of the density overlap matrix
    S_pos_mean: ArrayLike, posterior mean entropy estimated from the maximum likelihood states
    S_pos_se: ArrayLike, the estimated standard error of the entropy estimate
    eps: ArrayLike, the eps parameter used for DBSCAN
    min_probability_mass: ArrayLike, the minimum probability mass used in DBSCAN
    density_overlap: Array, the density overlap matrix
    states: Optional[ArrayLike]=None
    """

    C: Array
    C_core: Array
    C_non_noise: Array
    w: Array
    S: ArrayLike
    S_vn: ArrayLike
    S_pos_mean: ArrayLike
    S_pos_se: ArrayLike
    eps: ArrayLike
    min_probability_mass: ArrayLike
    density_overlap: Array


def neighborhood_matrix(D: Array, eps: ArrayLike) -> Array:
    A = 1 - jnp.heaviside(D - eps, 0)
    return A


def compute_distances(mixture: MixtureFitState) -> Array:
    cross_counts = vmap(lambda x: jnp.sqrt(x).T @ jnp.sqrt(x))(mixture.r)

    def normalize_cross_counts(x):
        return (
            jnp.sqrt(jnp.diag(1 / jnp.diag(x)))
            @ x
            @ jnp.sqrt(jnp.diag(1 / jnp.diag(x)))
        )

    cosine_distances = 1 - vmap(normalize_cross_counts)(cross_counts)
    return cosine_distances


def matrix_dbscan(
    D: Array, weights: Array, r: Array, eps: ArrayLike, min_probability_mass: ArrayLike
) -> CoarseGraining:
    """
    Vectorized DBSCAN applied to the distance matrix D for weighted samples.


    Parameters
    ----------
    D : the precomputed distance matrix, assumed to be symmetric and positive-definite.
    weights: the weight of each sample
    r : the responsibility of each mixture component for  each observation
    eps : the epsilon parameter for DBSCAN.
    min_probability mass : the minimum total weight of a point to be considered as a core sample

    Returns
    -------
    CoarseGraining
    """

    # note that this works as a coarse-graining matrix
    # because jnp.argmax will returns the first index
    # in the event of a tie
    N = neighborhood_matrix(D, eps)
    w_N = N.dot(weights)
    core_mask = jnp.heaviside(w_N - min_probability_mass, 1)
    core_and_gate = jnp.outer(core_mask, core_mask)
    # core adjacency matrix
    A_core = core_and_gate.at[jnp.diag_indices_from(core_and_gate)].set(0) * N
    # matrix exponential
    C_core = jnp.heaviside(jax.scipy.linalg.expm(A_core, max_squarings=48), 0)
    symmetrize = lambda X: X + X.T

    one_core_mask = symmetrize(jnp.outer(core_mask, 1 - core_mask))
    one_core_mask = one_core_mask.at[jnp.diag_indices_from(one_core_mask)].set(0)
    D_border_core = one_core_mask * D * N
    A_border_core = jnp.heaviside(one_core_mask * N, 0) * D

    # mask to keep only the minimum distance in the case of ambiguity
    A_border_core = jnp.int64(
        A_border_core == jnp.min(A_border_core + 1 * (A_border_core == 0), axis=0)
    )
    C_non_noise = C_core * core_mask.reshape(-1) + C_core.dot(
        A_border_core * (1 - core_mask.reshape(-1))
    )

    core_border_weights = jnp.dot(C_non_noise, weights)

    zero_weight_rows = 1 - jnp.heaviside(core_border_weights, 0)
    noise_mask = 1 - jnp.heaviside(C_non_noise.sum(axis=0), 0)

    C_all = C_non_noise + jnp.outer(zero_weight_rows, noise_mask)

    N_obs = r.shape[0]
    C_all = jnp.unique(C_all, axis=0, size=C_all.shape[0], fill_value=0)
    r_new = C_all.dot(r.T).T

    density_matrix = jnp.sqrt(r_new).T.dot(jnp.sqrt(r_new))
    density_matrix = density_matrix / jnp.trace(density_matrix)
    p_new = r_new.mean(axis=0)

    # gives the noise points a value of +1 and others -1 for the purposes of sorting by probability,
    # ensuring that noise points are the -1 entry in the array
    noise_sign = 2 * (jnp.heaviside(C_all.dot(noise_mask), 0) - 0.5)
    ix_sort = jnp.argsort(p_new * noise_sign)
    C_all = C_all[ix_sort, :]
    p_new = p_new[ix_sort]

    S = entr(p_new).sum()
    S_vn = ee.S_vn(density_matrix)
    S_pos = ee.S_posterior_mean_std(p_new * N_obs)
    return CoarseGraining(
        C_all,
        C_core,
        C_non_noise,
        p_new,
        S,
        S_vn,
        S_pos[0],
        S_pos[1],
        eps,
        min_probability_mass,
        density_matrix,
    )


def dbscan_jackknife(
    D: Array, weights: Array, r: Array, eps: float, min_probability_mass: float
):
    arr = jnp.arange(0, len(weights))

    def single_jackknife(i):
        rp = r.at[:, i].set(0)
        rp = rp / jnp.expand_dims(rp.sum(axis=1), -1)
        weightsp = rp.mean(axis=0)
        result = matrix_dbscan(D, weightsp, rp, eps, min_probability_mass)
        moments = ee.S_posterior_moments(result.w * len(r))
        return moments

    moments = vmap(single_jackknife)(arr)
    mu = jnp.mean(moments[:, 0])
    se = jnp.sqrt(jnp.mean(moments[:, 1]) - jnp.mean(moments[:, 0]) ** 2)
    return jnp.array([mu, se]).flatten()


def dbscan_resample_eps(
    D: Array,
    weights: Array,
    r: Array,
    eps: float,
    key: Array,
    min_probability_mass: float,
    N: int = 100,
):
    keys = jax.random.split(key, N)
    Z = vmap(jax.random.normal)(keys)

    def single_sample(z):
        eps_random = eps + z * (1 - eps) ** 2
        result = matrix_dbscan(D, weights, r, eps_random, min_probability_mass)
        moments = ee.S_posterior_moments(result.w * len(r))
        return moments

    moments = vmap(single_sample)(Z)
    mu = jnp.mean(moments[:, 0])
    se = jnp.sqrt(jnp.mean(moments[:, 1]) - jnp.mean(moments[:, 0]) ** 2)
    return jnp.array([mu, se]).flatten()


def dbscan_trace_eps(
    D: Array,
    weights: Array,
    r: Array,
    Eps: Array = jnp.linspace(0.01, 0.99, 99),
    min_probability_mass: float = 0.01,
    vmap_chunksize: int = 10,
):
    @jit
    def single_sample(eps):
        coarse_graining = matrix_dbscan(D, weights, r, eps, min_probability_mass)
        moments = ee.S_posterior_moments(coarse_graining.w * len(r))
        mu = moments[0]
        sd = jnp.sqrt(moments[1] - mu**2)
        Z_sq_vn = (
            jnp.subtract(coarse_graining.S_vn, coarse_graining.S) / (sd + 1e-9)
        ) ** 2
        return EntropyTrace(
            coarse_graining.S, coarse_graining.S_vn, mu, sd, Z_sq_vn, Eps
        )

    return cvmap(single_sample, chunk_size=vmap_chunksize)(Eps)
    # return #jax.lax.map(single_sample, Eps)


def dbscan_eps_std(
    D: Array,
    weights: Array,
    r: Array,
    k: ArrayLike = 1,
    min_probability_mass: float = 0.01,
) -> ResidueStates:
    """
    Apply DBSCAN to the distance matrix D with the eps parameter
    set as one minus the standard deviation of the distances.
    """
    eps = 1 - k * jnp.std(D[jnp.triu_indices_from(D)])
    dbs = matrix_dbscan(D, weights, r, eps, min_probability_mass)
    states_traj = jnp.argmax(dbs.C @ r.T, axis=0).T
    states, counts = jnp.unique(
        states_traj, size=D.shape[0], fill_value=-1, return_counts=True
    )
    n_states = jnp.sum(states != -1)
    S_mle = ee.S_mle(counts)
    N_obs = r.shape[0]
    _, se = ee.S_posterior_mean_std(counts)
    S_mle_dbs = ee.S_mle(dbs.w)
    return ResidueStates(
        n_states,
        S_mle,
        se,
        states_traj,
        counts,
        S_mle_dbs,
        dbs.w,
        eps,
        min_probability_mass,
        dbs.C,
    )


def _format_counts_vector(states: Array, counts: Array):
    counts_ = jnp.empty_like(states)
    counts_ = counts_.at[states].set(counts[states])
    last_ix = jnp.max(states)
    count_of_state = jnp.squeeze(counts[states == last_ix])
    counts_ = counts_.at[last_ix].set(count_of_state)
    return counts_, last_ix
