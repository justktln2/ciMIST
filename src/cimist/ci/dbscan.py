"""
DBSCAN operations for inferring conformations from von Mises mixture models
using density overlap.
"""

from typing import NamedTuple, Optional
from dataclasses import dataclass
import jax.numpy as jnp
import jax
from jax.scipy.special import entr
from jax import Array, jit, vmap
from jax.typing import ArrayLike
import cimist.entropy as ee
from cimist.utils.chunked_vmap import vmap_chunked as cvmap
from cimist.ci.vmm import MixtureFitState

@dataclass
class ResidueStates(NamedTuple):
    """
    Represents residue conformations, including trajectories of conformations,
    which are stored in the attribute states_traj.

    Attributes
    ----------
    n_states: ArrayLike, the number of states inferred by DBSAN
    S_mle: ArrayLike, the "plug-in" estimator of the entropy estimated from the maximum
    likelihood states.
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
    """
    Represents the entropy trace for a given epsilon value in DBSCAN.
    Attributes
    ----------
    S_mle: Array, the entropy estimated from the maximum likelihood states.
    S_vn: Array, the von Neumann entropy of the density overlap matrix.
    S_pos_mean: Array, posterior mean entropy estimated from the maximum likelihood states.
    S_pos_se: Array, the estimated standard error of the entropy estimate.
    Z_sq_vn: Array, the squared z-score of the von Neumann entropy.
    Eps: Array, the epsilon values used in DBSCAN.
    """

    S_mle: Array
    S_vn: Array
    S_pos_mean: Array
    S_pos_se: Array
    Z_sq_vn: Array
    Eps: Array


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
    S: ArrayLike, the "plug-in" estimator of the entropy estimated from the maximum likelihood
    states.
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


def neighborhood_matrix(d: Array, eps: ArrayLike) -> Array:
    """
    Compute the neighborhood matrix for DBSCAN.

    Parameters
    ----------
    d : Array
        The distance matrix, assumed to be symmetric and positive-definite.
    eps : ArrayLike
        The epsilon parameter for DBSCAN.

    Returns
    -------
    Array
        The neighborhood matrix, where each entry is 1 if the distance is less than or equal to eps,
        and 0 otherwise.
    """
    a = 1 - jnp.heaviside(d - eps, 0)
    return a


def compute_distances(mixture: MixtureFitState) -> Array:
    """
    Compute the distance matrix for the von Mises mixture components.

    Parameters
    ----------
    mixture : MixtureFitState
        The mixture fit state containing the responsibilities of each mixture component.

    Returns
    -------
    Array
        The distance matrix, where each entry is the cosine distance between the mixture
        components.
    """
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
    d: Array, weights: Array, r: Array, eps: ArrayLike, min_probability_mass: ArrayLike
) -> CoarseGraining:
    """
    Vectorized DBSCAN applied to the distance matrix D for weighted samples.


    Parameters
    ----------
    d : the precomputed distance matrix, assumed to be symmetric and positive-definite.
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
    n = neighborhood_matrix(d, eps)
    w_n = n.dot(weights)
    core_mask = jnp.heaviside(w_n - min_probability_mass, 1)
    core_and_gate = jnp.outer(core_mask, core_mask)
    # core adjacency matrix
    a_core = core_and_gate.at[jnp.diag_indices_from(core_and_gate)].set(0) * n
    # matrix exponential
    c_core = jnp.heaviside(jax.scipy.linalg.expm(a_core, max_squarings=48), 0)

    def symmetrize(X: Array) -> Array:
        return X + X.T

    one_core_mask = symmetrize(jnp.outer(core_mask, 1 - core_mask))
    one_core_mask = one_core_mask.at[jnp.diag_indices_from(one_core_mask)].set(0)
    a_border_core = jnp.heaviside(one_core_mask * n, 0) * d

    # mask to keep only the minimum distance in the case of ambiguity
    a_border_core = jnp.int64(
        a_border_core == jnp.min(a_border_core + 1 * (a_border_core == 0), axis=0)
    )
    c_non_noise = c_core * core_mask.reshape(-1) + c_core.dot(
        a_border_core * (1 - core_mask.reshape(-1))
    )

    core_border_weights = jnp.dot(c_non_noise, weights)

    zero_weight_rows = 1 - jnp.heaviside(core_border_weights, 0)
    noise_mask = 1 - jnp.heaviside(c_non_noise.sum(axis=0), 0)

    c_all = c_non_noise + jnp.outer(zero_weight_rows, noise_mask)

    n_obs = r.shape[0]
    c_all = jnp.unique(c_all, axis=0, size=c_all.shape[0], fill_value=0)
    r_new = c_all.dot(r.T).T

    density_matrix = jnp.sqrt(r_new).T.dot(jnp.sqrt(r_new))
    density_matrix = density_matrix / jnp.trace(density_matrix)
    p_new = r_new.mean(axis=0)

    # gives the noise points a value of +1 and others -1 for the purposes of sorting by probability,
    # ensuring that noise points are the -1 entry in the array
    noise_sign = 2 * (jnp.heaviside(c_all.dot(noise_mask), 0) - 0.5)
    ix_sort = jnp.argsort(p_new * noise_sign)
    c_all = c_all[ix_sort, :]
    p_new = p_new[ix_sort]

    s = entr(p_new).sum()
    s_vn = ee.S_vn(density_matrix)
    s_pos = ee.S_posterior_mean_std(p_new * n_obs)
    return CoarseGraining(
        c_all,
        c_core,
        c_non_noise,
        p_new,
        s,
        s_vn,
        s_pos[0],
        s_pos[1],
        eps,
        min_probability_mass,
        density_matrix,
    )


def dbscan_jackknife(
    d: Array, weights: Array, r: Array, eps: float, min_probability_mass: float
):
    """
    Perform jackknife resampling on the DBSCAN results.
        Parameters
        ----------
        d : Array
            The distance matrix, assumed to be symmetric and positive-definite.
        weights : Array
            The weight of each sample.
        r : Array
            The responsibility of each mixture component for each observation.
        eps : float
            The epsilon parameter for DBSCAN.
        min_probability_mass : float
            The minimum total weight of a point to be considered as a core sample.
        Returns
        -------
        Array
            The jackknife mean and standard error of the entropy estimate.
    """
    arr = jnp.arange(0, len(weights))

    def single_jackknife(i):
        rp = r.at[:, i].set(0)
        rp = rp / jnp.expand_dims(rp.sum(axis=1), -1)
        weightsp = rp.mean(axis=0)
        result = matrix_dbscan(d, weightsp, rp, eps, min_probability_mass)
        moments = ee.S_posterior_moments(result.w * len(r))
        return moments

    moments = vmap(single_jackknife)(arr)
    mu = jnp.mean(moments[:, 0])
    se = jnp.sqrt(jnp.mean(moments[:, 1]) - jnp.mean(moments[:, 0]) ** 2)
    return jnp.array([mu, se]).flatten()


def dbscan_resample_eps(
    d: Array,
    weights: Array,
    r: Array,
    eps: float,
    key: Array,
    min_probability_mass: float,
    n: int = 100,
):
    """
    Perform resampling on the DBSCAN results using random noise.
    Parameters
    ----------
    d : Array
        The distance matrix, assumed to be symmetric and positive-definite.
    weights : Array
        The weight of each sample.
    r : Array
        The responsibility of each mixture component for each observation.
    eps : float
        The epsilon parameter for DBSCAN.
    key : Array
        The JAX random key for generating random noise.
    min_probability_mass : float
        The minimum total weight of a point to be considered as a core sample.
    n : int, optional
        The number of resamples to perform. Default is 100.
    Returns
    -------
    Array
        The mean and standard error of the entropy estimate from the resampled
        DBSCAN results.
    """
    keys = jax.random.split(key, n)
    Z = vmap(jax.random.normal)(keys)

    def single_sample(z):
        eps_random = eps + z * (1 - eps) ** 2
        result = matrix_dbscan(d, weights, r, eps_random, min_probability_mass)
        moments = ee.S_posterior_moments(result.w * len(r))
        return moments

    moments = vmap(single_sample)(Z)
    mu = jnp.mean(moments[:, 0])
    se = jnp.sqrt(jnp.mean(moments[:, 1]) - jnp.mean(moments[:, 0]) ** 2)
    return jnp.array([mu, se]).flatten()


def dbscan_trace_eps(
    d: Array,
    weights: Array,
    r: Array,
    eps: Optional[Array] = jnp.linspace(0.01, 0.99, 99),
    min_probability_mass: Optional[float] = 0.01,
    vmap_chunksize: Optional[int] = 10,
):
    """
    Perform a trace of the DBSCAN results over a range of epsilon values.
    Parameters
    ----------
    d : Array
        The distance matrix, assumed to be symmetric and positive-definite.
    weights : Array
        The weight of each sample.
    r : Array
        The responsibility of each mixture component for each observation.
    eps : Array, optional
        The epsilon parameter for DBSCAN. Default is a linspace from 0.01 to 0.99.
    min_probability_mass : float, optional
        The minimum total weight of a point to be considered as a core sample.
        Default is 0.01.
    vmap_chunksize : int, optional
        The chunk size for vmap. Default is 10.
    Returns
    -------
    Array
        The entropy trace for the DBSCAN results over the range of epsilon values.
    """

    @jit
    def single_sample(eps):
        coarse_graining = matrix_dbscan(d, weights, r, eps, min_probability_mass)
        moments = ee.S_posterior_moments(coarse_graining.w * len(r))
        mu = moments[0]
        sd = jnp.sqrt(moments[1] - mu**2)
        z_sq_vn = ((coarse_graining.S_vn - coarse_graining.S) / (sd + 1e-9)) ** 2
        return EntropyTrace(
            coarse_graining.S, coarse_graining.S_vn, mu, sd, z_sq_vn, eps
        )

    return cvmap(single_sample, chunk_size=vmap_chunksize)(eps)
    # return #jax.lax.map(single_sample, eps)


def dbscan_eps_std(
    d: Array,
    weights: Array,
    r: Array,
    k: ArrayLike = 1,
    min_probability_mass: float = 0.01,
) -> ResidueStates:
    """
    Apply DBSCAN to the distance matrix d with the eps parameter
    set as one minus the standard deviation of the distances.
    Parameters
    ----------
    d : Array
        The distance matrix, assumed to be symmetric and positive-definite.
    weights : Array
        The weight of each sample.
    r : Array
        The responsibility of each mixture component for each observation.
    k : ArrayLike, optional
        The scaling factor for the epsilon parameter. Default is 1.
    min_probability_mass : float, optional
        The minimum total weight of a point to be considered as a core sample.
        Default is 0.01.
    Returns
    -------
    ResidueStates
        The inferred residue states, including the number of states, maximum likelihood
        entropy estimate, standard error of the entropy estimate, trajectory of states,
        counts of each state, entropy estimate from DBSCAN weights, DBSCAN weights,
        epsilon parameter used in DBSCAN, minimum probability mass used in DBSCAN,
        and the coarse-graining operator applied to the von Mises mixture components.
    """
    eps = 1 - k * jnp.std(d[jnp.triu_indices_from(d)])
    dbs = matrix_dbscan(d, weights, r, eps, min_probability_mass)
    states_traj = jnp.argmax(dbs.C @ r.T, axis=0).T
    states, counts = jnp.unique(
        states_traj, size=d.shape[0], fill_value=-1, return_counts=True
    )
    n_states = jnp.sum(states != -1)
    s_mle = ee.S_mle(counts)
    _, se = ee.S_posterior_mean_std(counts)
    s_mle_dbs = ee.S_mle(dbs.w)
    return ResidueStates(
        n_states,
        s_mle,
        se,
        states_traj,
        counts,
        s_mle_dbs,
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
