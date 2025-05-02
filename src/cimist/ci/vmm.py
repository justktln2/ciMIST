"""
Utilities for von Mises mixture models.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, jit, vmap
import jax
from jax.lax import cond, fori_loop
from jax.scipy.special import logsumexp, i0, xlogy
from jax.nn import softmax, log_softmax
from jax.scipy.optimize import minimize


class MixtureFit(NamedTuple):
    n_components: Array
    mu: Array
    kappa: Array
    logw: Array
    mask: Array
    r: Array
    log_likelihood: Array
    n_iter: Array
    converged: Array
    atol: Array
    statuses: Array
    njevs: Array
    nfevs: Array
    nits: Array
    successes: Array


class MixtureFitState(NamedTuple):
    n_components: int
    mu: Array
    kappa: Array
    logw: Array
    mask: Array
    r: Array
    log_likelihood: Array
    n_iter: int
    converged: Array
    atol: float
    statuses: Array
    njevs: Array
    nfevs: Array
    nits: Array
    successes: Array


def sort_by_weight(state: MixtureFitState) -> MixtureFitState:
    ix = jnp.argsort(-state.logw)  # get index from highest to lowest probability
    state = state._replace(
        mu=state.mu[ix], kappa=state.kappa[ix], logw=state.logw[ix], r=state.r[:, ix]
    )
    return state


def von_mises_mixture_log_pdf(
    x: Array, mu: Array, kappa: Array, logw: Array, mask: Array
):
    """
    log-pdf of the von mises fisher mixture distribution
    """
    # January 18, 2023: checked that this agrees with implementation in mask.py
    # log-likelihoods of individual components
    minus_FE = vmap(
        lambda mu, kappa, logw: von_mises_component_log_prob(x, mu, kappa, logw, mask),
        in_axes=(0, 0, 0),
    )
    # aggregate over all, ensuring logw is normalized
    return logsumexp(minus_FE(mu, kappa, log_softmax(logw)), axis=0)


def von_mises_component_log_prob(
    x: Array, mu: Array, kappa: Array, logw: Array, mask: Array
):
    """
    log-probability of an individual mixture component, with component weight
    factored in
    """
    return von_mises_log_pdf(x, mu, kappa, mask) + logw


def von_mises_log_pdf(x: Array, mu: Array, kappa: Array, mask: Array):
    summands = jnp.where(
        mask, kappa * jnp.cos(x - mu) - jnp.log(2 * jnp.pi * i0(kappa)), 0.0
    )
    return jnp.sum(summands, axis=-1)


def von_mises_pdf(x: Array, mu: Array, kappa: Array, mask: Array):
    log_prob = von_mises_log_pdf(x, mu, kappa, mask)
    max_log_prob = jnp.max(log_prob)
    return jnp.exp(log_prob - max_log_prob) * jnp.exp(max_log_prob)


def von_mises_mixture_pdf(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    """
    pdf of the von mises fisher mixture distribution
    """
    log_prob = von_mises_mixture_log_pdf(x, mu, kappa, logw, mask)
    max_log_prob = jnp.max(log_prob)
    return jnp.exp(log_prob - max_log_prob) * jnp.exp(max_log_prob)


def e_step(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    component_log_likelihoods = vmap(
        lambda mu, kappa: von_mises_log_pdf(x, mu, kappa, mask),
        in_axes=(0, 0),
        out_axes=1,
    )
    r = softmax(component_log_likelihoods(mu, kappa) + logw, axis=1)
    return r


def expected_log_likelihood(
    x: Array, mu: Array, kappa: Array, r: Array, w: Array, mask: Array
):
    # expected log likelihood of the mixture model over the posterior distribution of the
    # hidden labels
    component_log_likelihoods = vmap(
        lambda mu, kappa: von_mises_log_pdf(x, mu, kappa, mask),
        in_axes=(0, 0),
        out_axes=1,
    )
    return jnp.sum(r * component_log_likelihoods(mu, kappa)) + jnp.sum(xlogy(w, w))


def mixture_log_likelihood(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    return jnp.mean(von_mises_mixture_log_pdf(x, mu, kappa, logw, mask))


def mu_mle(theta: Array, r: Array):
    """
    Mean angle given weights in the array r.
    r need not be normalized.
    """
    complex_exp = jax.lax.complex(jnp.cos(theta), jnp.sin(theta))
    num = r.T @ complex_exp
    denom = jnp.sqrt(jnp.real(num * num.conjugate()))
    return jnp.where(denom > 0, jnp.angle(num / denom), 0)


def m_step(
    theta: Array,
    mu: Array,
    kappa: Array,
    r: Array,
    mask: Array,
    gtol=1e-3,
    maxiter=100,
    line_search_maxiter=10,
    min_kappa=10.0,
    max_kappa=600.0,
):
    # helper functions
    def map_to_mpi_pi(x):
        x_new = jnp.mod(x, 2 * jnp.pi)
        return jnp.where(mask, x_new - 2 * jnp.pi * (x_new > jnp.pi), 0)

    def scale_kappa(kappa):
        return jnp.where(mask, jnp.minimum(jnp.maximum(min_kappa, kappa), max_kappa), 0)

    M = theta.shape[0]
    w = r.sum(axis=0)

    # Get the closed-form maximum expected log likelihood estimate for mu
    mu = mu_mle(theta, r)

    # define the loss function for kappa and optimize using bfgs
    def negative_ell(p):
        # expected log likelihood of the kappa parameters
        _p = p.reshape(kappa.shape)
        kappas = scale_kappa(_p)
        # log likelihood per observation
        return -expected_log_likelihood(theta, mu, kappas, r, w, mask) / M

    x0 = kappa.flatten()
    solver_options = dict(
        gtol=gtol, maxiter=maxiter, line_search_maxiter=line_search_maxiter
    )
    result = minimize(negative_ell, x0, method="bfgs", options=solver_options)
    new_kappa = result.x.reshape(kappa.shape)
    kappa = jnp.where(
        jnp.isfinite(new_kappa), new_kappa, kappa
    )  # get rid of nans and infs from the solver

    return map_to_mpi_pi(mu), scale_kappa(kappa), jnp.log(w / w.max()), result


def advance_em(
    theta: Array, state: MixtureFitState, steps: int = 10, gtol=1e-3, gmaxiter=500
) -> MixtureFitState:
    """
    Advance the EM algorithm by n_iter steps, starting from the given state.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    state : TYPE
        DESCRIPTION.
    n_iter : TYPE, optional
        DESCRIPTION. The default is 10.
    atol : TYPE, optional
        DESCRIPTION. The default is 1e-3.
    gtol : TYPE, optional
        DESCRIPTION. The default is 1e-2.
    gmaxiter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    state : TYPE
        DESCRIPTION.

    """

    # advance if converged, only using the for loop if steps>1, return current state otherwise
    state = cond(
        state.converged,
        lambda: state,
        lambda: fori_loop(0, steps, lambda i, s: em_step(i, theta, s), state),
    )

    return sort_by_weight(state)


def is_converged(state: MixtureFitState) -> bool:
    converged = cond(
        state.n_iter > 0,
        lambda x: jnp.abs(x.log_likelihood[x.n_iter] - x.log_likelihood[x.n_iter - 1])
        < x.atol,
        lambda x: False,
        state,
    )
    return converged


@jit
def update_r(theta: Array, state: MixtureFitState) -> MixtureFitState:
    r = e_step(theta, state.mu, state.kappa, state.logw, state.mask)
    state = state._replace(r=r)
    return sort_by_weight(state)


def predict_proba(theta: Array, state: MixtureFitState) -> Array:
    return e_step(theta, state.mu, state.kappa, state.logw, state.mask)


@jit
def em_step(
    theta: Array, state: MixtureFitState, gtol=1e-3, gmaxiter=500
) -> MixtureFitState:
    r = predict_proba(theta, state)
    mu, kappa, logw, result = m_step(
        theta, state.mu, state.kappa, r, state.mask, gtol=gtol, maxiter=gmaxiter
    )
    ll = mixture_log_likelihood(theta, state.mu, state.kappa, state.logw, state.mask)
    log_likelihood = state.log_likelihood.at[state.n_iter].set(ll)
    state = state._replace(
        mu=mu,
        kappa=kappa,
        logw=logw,
        log_likelihood=log_likelihood,
        r=r,
        nfevs=state.nfevs.at[state.n_iter].set(result.nfev),
        njevs=state.nfevs.at[state.n_iter].set(result.njev),
        nits=state.nits.at[state.n_iter].set(result.nit),
        successes=state.successes.at[state.n_iter].set(result.success),
        statuses=state.statuses.at[state.n_iter].set(result.status),
    )

    converged = is_converged(state)

    state = state._replace(
        n_iter=state.n_iter + 1,
        converged=converged,  # type: ignore
    )
    return state


def step_if_not_converged(
    theta: Array, state: MixtureFitState, gtol=1e-3, gmaxiter=500
):
    def do_nothing(theta, state, gtol=1e-3, gmaxiter=500):
        return state

    state = cond(state.converged, do_nothing, em_step, theta, state, gtol, gmaxiter)
    return state


def init_random_mixture_state(
    theta, mask, key, n_components=100, maxiter=100, atol=1e-2
):
    # initialize random mixture states using kmeans++ to set the cluster centers and
    # random concentration parameters
    key_kmpp, key_kappa = jax.random.split(key)
    # set mus
    mu = spherical_kmeans_plus_plus(theta, n_components, key_kmpp)
    params = _init_params(mu, mask, n_components, key_kappa)
    log_likelihood = jnp.empty(maxiter)
    statuses = jnp.empty_like(log_likelihood)
    njevs = jnp.empty_like(log_likelihood)
    nfevs = jnp.empty_like(log_likelihood)
    nits = jnp.empty_like(log_likelihood)
    successes = jnp.empty_like(log_likelihood)
    statuses = jnp.empty_like(log_likelihood)

    state = MixtureFitState(
        n_components=n_components,
        mu=params["mu"],
        kappa=params["kappa"],
        logw=params["logw"],
        mask=mask,
        r=jnp.empty((theta.shape[0], n_components)),
        log_likelihood=log_likelihood,
        n_iter=0,
        converged=False,  # type: ignore
        atol=atol,
        statuses=statuses,
        njevs=njevs,
        nfevs=nfevs,
        nits=nits,
        successes=successes,
    )
    return state


def warm_start(theta, mixture_fit, mask, key, n_components=100, maxiter=100, atol=1e-2):
    # initialize random mixture states using kmeans++ to set the cluster centers and
    # random concentration parameters
    key_kmpp, key_kappa = jax.random.split(key)
    # set mus
    mu = spherical_kmeans_plus_plus(theta, n_components, key_kmpp)
    params = _init_params(mu, mask, n_components, key_kappa)
    log_likelihood = jnp.empty(maxiter)
    statuses = jnp.empty_like(log_likelihood)
    njevs = jnp.empty_like(log_likelihood)
    nfevs = jnp.empty_like(log_likelihood)
    nits = jnp.empty_like(log_likelihood)
    successes = jnp.empty_like(log_likelihood)
    statuses = jnp.empty_like(log_likelihood)

    state = MixtureFitState(
        n_components=n_components,
        mu=params["mu"],
        kappa=params["kappa"],
        logw=params["logw"],
        mask=mask,
        r=jnp.empty((theta.shape[0], n_components)),
        log_likelihood=log_likelihood,
        n_iter=0,
        converged=False,  # type: ignore
        atol=atol,
        statuses=statuses,
        njevs=njevs,
        nfevs=nfevs,
        nits=nits,
        successes=successes,
    )
    return state


def spherical_kmeans_plus_plus(theta, n_clusters, key):
    # adaptation of kmeans plus_plus to the unit sphere
    """
    Note that for this, the cluster centers are initialized as *data points* so
    it should work reasonably well for our purposes. No need to calculate any
    kind of average.
    The concentration parameters can probably be set in the same way as I was setting
    them before, using heuristics about how well I expect angles to be localized.

    The exact algorithm is as follows:
    1. Choose one center uniformly at random among the data points.

    2. For each data point x not chosen yet, compute D(x),
    the distance between x and the nearest center that has already been chosen.

    3. Choose one new data point at random as a new center,
    using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.

    4. Repeat Steps 2 and 3 until k centers have been chosen.
    5. Now that the initial centers have been chosen, proceed using standard k-means clustering.

    """
    keys = jax.random.split(key, n_clusters)  # 100 x 2 array
    ix = jax.random.randint(keys[0], (1,), 0, theta.shape[0])

    inv_norm = 1 / jnp.linalg.norm(theta, axis=1)
    # theta_hat = jnp.diag(1/norm) @ theta # SAMPLES x 28, THIS CAUSES MEMORY PROBLEMS
    theta_hat = theta * inv_norm[:, jnp.newaxis]

    normalized_cluster_centers = theta_hat[ix].T
    ixs = [ix]

    for k in keys[1:]:
        # compute cosine distances to cluster centers
        cosine_distances = 1 - theta_hat @ normalized_cluster_centers

        # probability of choosing cluster center is proportional to min_distance**2
        min_distance = jnp.min(cosine_distances, axis=1)
        log_p = jnp.where(
            min_distance > 1e-8,
            2 * jnp.log(min_distance),
            -jnp.inf,  # probability of zero for something already chosen
        )  # (un-normalized) log probability of choosing index
        ix = jax.random.categorical(k, log_p, shape=(1,))

        new_center = theta_hat[ix].T

        normalized_cluster_centers = jnp.hstack(
            (normalized_cluster_centers, new_center)
        ).squeeze()
        ixs.append(ix)

    ixs = jnp.concatenate(ixs)
    centers = theta[ixs]
    return centers


def _init_params(
    mu: Array,
    cloud_mask: Array,
    K: int,
    key=jax.random.PRNGKey(0),
):
    """
    Assumes mu has been generated with kmeans++. These need to be separate because
    running kmeans++ with vmap tends to blow out the memory on 256GB machines for
    reasonably-sized trajectories.

    initialize random concentration parameters from a mask
    that sets the number of features (by zeroing out everything else)
    and an integer that sets the number of components
    """
    w = jnp.ones(K) / K

    kappa_bond_key, kappa_torsion_key = jax.random.split(key, 2)

    # kappas = 25*jnp.ones((K,len(cloud_mask))) * cloud_mask # localize to within about 30 degrees in each direction
    def concat(*args):
        return jnp.concatenate(args, axis=-1)

    # initialize mu parameters to account for stronger forces constraining bond angles
    # and omega torsion angle
    # mu_bond = 1.0 + 0.05*jax.random.normal(mu_bond_key, shape=(K, 14)) * cloud_mask[:14]
    # mu_omega = jnp.pi + 0.1 * jax.random.normal(mu_omega_key, shape=(K,1)) * cloud_mask[14]
    # mu_torsion = jnp.array(jax.random.uniform(
    #        mu_torsion_key,
    #        minval=-jnp.pi, maxval=jnp.pi,
    #        shape=(K, 13)
    #    )) * cloud_mask[15:]
    # mu = concat(mu_bond, mu_omega, mu_torsion)

    # initialize the bond angles to be more tightly constrained than torsion angles
    # kappa_bond is actually bond angles plus omega torsion, which is tightly constrained
    # at 180 degrees
    kappa_bond = (
        jax.random.uniform(kappa_bond_key, minval=100, maxval=500, shape=(K, 15))
        * cloud_mask[:15]
    )  # bond and omega torsion
    # kappa_torsion excludes omega torsion, which is localized like a bond angle
    # torsion mixture components should localize to between about 10 and 30 degrees
    kappa_torsion = (
        jax.random.uniform(kappa_torsion_key, minval=25, maxval=50, shape=(K, 13))
        * cloud_mask[15:]
    )  # torsion

    kappa = concat(kappa_bond, kappa_torsion)
    params = dict(mu=mu, kappa=kappa, logw=jax.nn.log_softmax(jnp.log(w)))
    return params


@jit
def analytical_kl(mu, kappa, mask):
    # uses analytical expression for KL-divergence from Kitagawa and Rowley
    i1 = vmap(jax.grad(i0))
    mu_vecs = jnp.stack([jnp.cos(mu), jnp.sin(mu)])
    # 0*(jnp.log(kappa[0] / kappa[1])),
    kappa_terms = jnp.where(mask, -jnp.log(i0(kappa[0]) / i0(kappa[1])), 0.0)
    rv = i1(kappa[0]) / i0(
        kappa[0]
    )  # *(kappa[0]*complex_mu[0] - kappa[1])#*complex_mu[1])
    # only need on diagonal terms
    # via equation 37 of K&R this should also be equal to rv * (kappa[0] - kappa[1]*jnp.cos(mu[0]-mu[1]))
    mu_terms = jnp.diag(
        (rv * (kappa[0] * mu_vecs[:, 0, :] - kappa[1] * mu_vecs[:, 1, :])).T.dot(
            mu_vecs[:, 0]
        )
    )
    return mu_terms.sum() + kappa_terms.sum()


def load_mixture_state(f):
    states_loaded = jnp.load(f)
    return MixtureFitState(
        n_components=int(states_loaded["n_components"]),
        mu=jnp.array(states_loaded["mu"]),
        kappa=jnp.array(states_loaded["kappa"]),
        logw=jnp.array(states_loaded["logw"]),
        mask=jnp.array(states_loaded["mask"]),
        r=jnp.array(states_loaded["r"]),
        log_likelihood=jnp.array(states_loaded["log_likelihood"]),
        n_iter=int(states_loaded["n_iter"]),
        converged=bool(states_loaded["converged"]),  # type: ignore
        atol=float(states_loaded["atol"]),
        statuses=jnp.array(states_loaded["statuses"]),
        njevs=jnp.array(states_loaded["njevs"]),
        nfevs=jnp.array(states_loaded["nfevs"]),
        nits=jnp.array(states_loaded["nits"]),
        successes=jnp.array(states_loaded["successes"]),
    )
