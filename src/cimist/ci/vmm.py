"""
Utilities for von Mises mixture models.
"""
import numpy as np
import scipy.linalg #needs to be here for now for ridiculous install problem reasons

from typing import NamedTuple, Optional, Union
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, vmap
import jax
from jax.lax import cond, fori_loop
from jax.scipy.special import logsumexp, i0, xlogy, i0e, i1e
from jax.nn import softmax, log_softmax
from jax.scipy.optimize import minimize

class ScalarSolveResult(NamedTuple):
    success: Array
    status: Array
    nit: Array
    nfev: Array
    njev: Array
    roundoff_clips: Array


class MixtureFitState(NamedTuple):
         n_components: int
         mu: Array
         kappa: Array
         logw: Array
         mask: Array
         r: Array
         log_likelihood: Array
         n_iter: int
         converged: bool
         atol: float
         statuses: Array
         njevs: Array
         nfevs: Array
         nits: Array
         successes: Array

def sort_by_weight(state: MixtureFitState) -> MixtureFitState:
    ix = jnp.argsort(-state.logw) # get index from highest to lowest probability
    state = state._replace(
        mu = state.mu[ix],
        kappa = state.kappa[ix],
        logw = state.logw[ix],
        r = state.r[:,ix]
    )
    return state

def von_mises_mixture_log_pdf(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    """
    log-pdf of the von mises fisher mixture distribution
    """
    # January 18, 2023: checked that this agrees with implementation in mask.py
    # log-likelihoods of individual components
    minus_FE = vmap(
        lambda mu, kappa, logw:
    von_mises_component_log_prob(x, mu, kappa, logw, mask), in_axes=(0, 0, 0)
    )
    # aggregate over all, ensuring logw is normalized
    return logsumexp(minus_FE(mu, kappa, log_softmax(logw)), axis=0)

def von_mises_component_log_prob(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    """
    log-probability of an individual mixture component, with component weight
    factored in
    """
    return von_mises_log_pdf(x, mu, kappa, mask) + logw

def von_mises_log_pdf(x: Array, mu: Array, kappa: Array, mask: Array):
    summands = jnp.where(mask, kappa*jnp.cos(x-mu) - jnp.log(2*jnp.pi*i0(kappa)), 0.0)
    return jnp.sum(summands, axis=-1)

def von_mises_pdf(x: Array, mu: Array, kappa: Array, mask: Array):
    log_prob = von_mises_log_pdf(x, mu, kappa, mask)
    max_log_prob = jnp.max(log_prob)
    return jnp.exp(log_prob - max_log_prob) * jnp.exp(max_log_prob)

def von_mises_mixture_pdf(x: Array,
                    mu: Array,
                    kappa: Array,
                    logw: Array,
                    mask: Array
                    ):
    """
    pdf of the von mises fisher mixture distribution
    """
    log_prob = von_mises_mixture_log_pdf(x, mu, kappa, logw, mask)
    max_log_prob = jnp.max(log_prob)
    return jnp.exp(log_prob - max_log_prob) * jnp.exp(max_log_prob)

def e_step(x: Array, mu: Array, kappa: Array, logw: Array, mask: Array):
    component_log_likelihoods = vmap(
        lambda mu, kappa: von_mises_log_pdf(x, mu, kappa, mask),
        in_axes=(0,0), out_axes=1
        )
    r = softmax(component_log_likelihoods(mu, kappa) + logw, axis=1)
    return r

def expected_log_likelihood(x: Array, mu: Array, kappa: Array, r: Array, w: Array, mask: Array):
    # expected log likelihood of the mixture model over the posterior distribution of the 
    # hidden labels
    component_log_likelihoods = vmap(
        lambda mu, kappa: von_mises_log_pdf(x, mu, kappa, mask),
        in_axes=(0,0), out_axes=1
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

def _m_step_bfgs(
    theta: Array,
    mu: Array,
    kappa: Array,
    r: Array,
    mask: Array,
    gtol=1e-3,
    maxiter=100,
    line_search_maxiter=10,
    min_kappa=10.0,
    max_kappa=600.0):

    # helper functions
    def map_to_mpi_pi(x):
        x_new = jnp.mod(x, 2*jnp.pi)
        return jnp.where(mask, x_new - 2*jnp.pi*(x_new > jnp.pi), 0)

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
    solver_options = dict(gtol=gtol, maxiter=maxiter, line_search_maxiter=line_search_maxiter)
    result = minimize(negative_ell, x0, method="bfgs", options=solver_options)
    new_kappa = result.x.reshape(kappa.shape)
    kappa = jnp.where(jnp.isfinite(new_kappa), new_kappa, kappa) # get rid of nans and infs from the solver

    return map_to_mpi_pi(mu), scale_kappa(kappa), jnp.log(w / w.max()), result

def _m_step_newton(
    theta: Array,
    mu: Array,
    kappa: Array,
    r: Array,
    mask: Array,
    gtol=1e-3,
    maxiter=100,
    line_search_maxiter=10,
    min_kappa=10.0,
    max_kappa=600.0):

    """Solve the separable S1 concentration M-step within the original bounds.

    Only small component-by-angle arrays enter the root iterations. Statuses:
    0 success, 1 iteration limit, 2 invalid input, 3 accepted resultant roundoff.
    ``gtol`` and ``line_search_maxiter`` apply only to the BFGS solver.
    """
    if theta.dtype != r.dtype or theta.dtype != kappa.dtype or theta.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("Newton theta, responsibilities and concentrations must share float32 or float64 dtype")
    # helper functions
    def map_to_mpi_pi(x):
        x_new = jnp.mod(x, 2*jnp.pi)
        return jnp.where(mask, x_new - 2*jnp.pi*(x_new > jnp.pi), 0)

    def scale_kappa(kappa):
        return jnp.where(mask, jnp.minimum(jnp.maximum(min_kappa, kappa), max_kappa), 0)

    w = r.sum(axis=0)

    # Get the closed-form maximum expected log likelihood estimate for mu
    mu = mu_mle(theta, r)

    # Same separable bounded objective; tighter solver changes the EM path.
    C = (r.T @ jnp.cos(theta)) * jnp.cos(mu) + (r.T @ jnp.sin(theta)) * jnp.sin(mu)
    active = jnp.broadcast_to(mask, kappa.shape) & (w[:, None] > 0)
    R = C / jnp.where(w[:, None] > 0, w[:, None], 1.)
    finite = (jnp.isfinite(theta).all() & jnp.isfinite(r).all()
              & jnp.isfinite(kappa).all() & (r >= 0).all()
              & jnp.isfinite(R).all() & jnp.isfinite(w).all() & (w.sum() > 0)
              & jnp.isfinite(min_kappa) & jnp.isfinite(max_kappa)
              & (min_kappa > 0) & (max_kappa >= min_kappa))
    invalid = ~finite | jnp.any(active & (jnp.abs(R) > 1 + 64 * jnp.finfo(R.dtype).eps))
    roundoff = jnp.sum(active & (jnp.abs(R) > 1))
    R = jnp.clip(R, -1., 1.)

    def A(x):
        return i1e(x) / i0e(x)

    lower = active & (R <= A(min_kappa))
    upper = active & (R >= A(max_kappa))
    interior = active & ~lower & ~upper
    x = jnp.clip(kappa, min_kappa, max_kappa)
    lo, hi = jnp.full_like(x, min_kappa), jnp.full_like(x, max_kappa)

    # Float64 retains the validated reference tolerances; float32 needs attainable ones.
    eps = jnp.finfo(R.dtype).eps
    root_tol = jnp.maximum(3e-15, 8 * eps)

    def residual(x, lo, hi):
        width_tol = jnp.maximum(1e-10, 8 * eps * jnp.abs(x))
        return interior & (jnp.abs(A(x) - R) > root_tol) & (hi - lo > width_tol)

    def condition(state):
        iteration, x, lo, hi = state
        return (iteration < jnp.minimum(maxiter, 100)) & jnp.any(residual(x, lo, hi)) & ~invalid

    def update(state):
        iteration, x, lo, hi = state
        a = A(x)
        f = a - R
        todo = residual(x, lo, hi)
        lo = jnp.where(todo & (f < 0), x, lo)
        hi = jnp.where(todo & (f >= 0), x, hi)
        derivative = 1 - a / x - a * a
        step = x - f / derivative
        step = jnp.where((derivative > 0) & jnp.isfinite(step) & (step > lo) & (step < hi),
                         step, (lo + hi) / 2)
        return iteration + 1, jnp.where(todo, step, x), lo, hi

    nit, x, lo, hi = jax.lax.while_loop(condition, update, (jnp.array(0), x, lo, hi))
    failed = jnp.any(residual(x, lo, hi)) | invalid
    solved = jnp.where(lower, min_kappa, jnp.where(upper, max_kappa, x))
    solved = jnp.where(failed, kappa, solved)
    # Status 3 records accepted roundoff above unit resultant; 2 rejects invalid input.
    status = jnp.where(invalid, 2, jnp.where(failed, 1, jnp.where(roundoff > 0, 3, 0)))
    result = ScalarSolveResult(~failed, status, nit, nit + 1, nit, roundoff)
    return map_to_mpi_pi(mu), scale_kappa(solved), jnp.log(w / w.max()), result

def m_step(theta: Array, mu: Array, kappa: Array, r: Array, mask: Array,
           gtol=1e-3, maxiter=100, line_search_maxiter=10,
           min_kappa=10.0, max_kappa=600.0, solver="newton"):
    """Update means, concentrations and weights; make ``solver`` static under JIT.

    ``bfgs`` preserves the original clipped-objective implementation. ``newton``
    solves its bounded concentration objective more tightly and can change fits.
    Both return optimizer diagnostics as the fourth result; inspect ``success``.
    """
    if solver == "newton":
        solve = _m_step_newton
    elif solver == "bfgs":
        solve = _m_step_bfgs
    else:
        raise ValueError("solver must be 'newton' or 'bfgs'")
    return solve(theta, mu, kappa, r, mask, gtol, maxiter,
                 line_search_maxiter, min_kappa, max_kappa)


def advance_em(theta: Array, state: MixtureFitState, steps: int=10,
               gtol=1e-3, gmaxiter=500, solver="newton") -> MixtureFitState:
    """Advance an unconverged state by at most ``steps`` updates, then sort by weight.

    ``solver`` must be static when this function is used under JIT.
    """
    state = cond(state.converged,
                 lambda: state,
                 lambda: fori_loop(
                     0, steps,
                     lambda i, s: step_if_not_converged(theta, s, gtol, gmaxiter, solver=solver),
                     state))
    return sort_by_weight(state)


def is_converged(state: MixtureFitState) -> bool:
    """Compare the latest two post-update likelihoods, once both exist."""
    converged = cond(state.n_iter > 1,
    lambda x: jnp.abs(x.log_likelihood[x.n_iter-1] - x.log_likelihood[x.n_iter-2])<x.atol,
    lambda x: False,
    state)
    return converged

@jit
def update_r(theta: Array, state: MixtureFitState) -> MixtureFitState:
    r = e_step(theta, state.mu, state.kappa, state.logw, state.mask)
    state = state._replace(r = r)
    return sort_by_weight(state)

def predict_proba(theta: Array, state: MixtureFitState) -> Array:
    return e_step(theta, state.mu, state.kappa, state.logw, state.mask)

@partial(jit, static_argnames=("solver",))
def em_step(theta: Array, state: MixtureFitState, gtol=1e-3, gmaxiter=500,
            solver="newton") -> MixtureFitState:
    r = predict_proba(theta, state)
    mu, kappa, logw, result = m_step(theta, state.mu, state.kappa, r, state.mask, gtol=gtol, maxiter=gmaxiter, solver=solver)
    # Score the parameters being returned, so convergence cannot lag an M-step.
    ll = mixture_log_likelihood(theta, mu, kappa, logw, state.mask)
    log_likelihood = state.log_likelihood.at[state.n_iter].set(ll)
    state = state._replace(
        mu = mu,
        kappa = kappa,
        logw = logw,
        log_likelihood = log_likelihood,
        r = r,
        nfevs = state.nfevs.at[state.n_iter].set(result.nfev),
        # Keep the historical BFGS diagnostics for reproducible baseline output.
        njevs = (state.nfevs if solver == "bfgs" else state.njevs).at[state.n_iter].set(result.njev),
        nits = state.nits.at[state.n_iter].set(result.nit),
        successes = state.successes.at[state.n_iter].set(result.success),
        statuses = state.statuses.at[state.n_iter].set(result.status),
        n_iter = state.n_iter + 1
    )
    
    converged = is_converged(state)
    
    state = state._replace(
        converged = converged
        )
    return state
    
def step_if_not_converged(theta: Array, state: MixtureFitState, gtol=1e-3,
                          gmaxiter=500, solver="newton"):
    """Apply one EM step; make ``solver`` static under JIT."""
    return cond(state.converged,
                lambda: state,
                lambda: em_step(theta, state, gtol, gmaxiter, solver=solver))


def init_random_mixture_state(theta, mask, key, n_components=100, maxiter=100, atol=1e-2):
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

    state = MixtureFitState(n_components=n_components,
                            mu=params["mu"],
                            kappa=params["kappa"],
                            logw=params["logw"],
                            mask=mask,
                            r=jnp.empty((theta.shape[0], n_components)),
                            log_likelihood=log_likelihood,
                            n_iter=0,
                            converged=False,
                            atol=atol,
                            statuses=statuses,
                            njevs=njevs,
                            nfevs=nfevs,
                            nits=nits,
                            successes=successes
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

    state = MixtureFitState(n_components=n_components,
                            mu=params["mu"],
                            kappa=params["kappa"],
                            logw=params["logw"],
                            mask=mask,
                            r=jnp.empty((theta.shape[0], n_components)),
                            log_likelihood=log_likelihood,
                            n_iter=0,
                            converged=False,
                            atol=atol,
                            statuses=statuses,
                            njevs=njevs,
                            nfevs=nfevs,
                            nits=nits,
                            successes=successes
                            )
    return state


@partial(jit, static_argnames=("n_clusters",))
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
    keys = jax.random.split(key, n_clusters) # 100 x 2 array
    first = jax.random.randint(keys[0], (1,), 0, theta.shape[0])[0]
    
    inv_norm = 1/jnp.linalg.norm(theta, axis=1)
    #theta_hat = jnp.diag(1/norm) @ theta # SAMPLES x 28, THIS CAUSES MEMORY PROBLEMS
    theta_hat = theta * inv_norm[:, jnp.newaxis]

    nearest = 1 - theta_hat @ theta_hat[first]

    def add_center(nearest, draw_key):
        log_p = jnp.where(nearest > 1e-8, 2*jnp.log(nearest), -jnp.inf)
        ix = jax.random.categorical(draw_key, log_p, shape=(1,))[0]
        # The old centers' minimum is unchanged when one new center is added.
        nearest = jnp.minimum(nearest, 1 - theta_hat @ theta_hat[ix])
        return nearest, ix

    _, remaining = jax.lax.scan(add_center, nearest, keys[1:])
    ixs = jnp.concatenate((first[None], remaining))
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
    #kappas = 25*jnp.ones((K,len(cloud_mask))) * cloud_mask # localize to within about 30 degrees in each direction
    concat = lambda *args: jnp.concatenate(args, axis=-1)

    # initialize mu parameters to account for stronger forces constraining bond angles
    # and omega torsion angle
    #mu_bond = 1.0 + 0.05*jax.random.normal(mu_bond_key, shape=(K, 14)) * cloud_mask[:14]
    #mu_omega = jnp.pi + 0.1 * jax.random.normal(mu_omega_key, shape=(K,1)) * cloud_mask[14]
    #mu_torsion = jnp.array(jax.random.uniform(
    #        mu_torsion_key,
    #        minval=-jnp.pi, maxval=jnp.pi,
    #        shape=(K, 13)
    #    )) * cloud_mask[15:]
    #mu = concat(mu_bond, mu_omega, mu_torsion)

    # initialize the bond angles to be more tightly constrained than torsion angles
    # kappa_bond is actually bond angles plus omega torsion, which is tightly constrained
    # at 180 degrees
    kappa_bond = jax.random.uniform(kappa_bond_key, minval=100, maxval=500, shape=(K, 15)) * cloud_mask[:15] # bond and omega torsion
    # kappa_torsion excludes omega torsion, which is localized like a bond angle
    # torsion mixture components should localize to between about 10 and 30 degrees
    kappa_torsion = jax.random.uniform(kappa_torsion_key, minval=25, maxval=50, shape=(K, 13)) * cloud_mask[15:] # torsion

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
    rv = i1(kappa[0])/i0(kappa[0])#*(kappa[0]*complex_mu[0] - kappa[1])#*complex_mu[1])
    # only need on diagonal terms
    # via equation 37 of K&R this should also be equal to rv * (kappa[0] - kappa[1]*jnp.cos(mu[0]-mu[1]))
    mu_terms = jnp.diag((rv*(kappa[0]*mu_vecs[:,0,:] - kappa[1]*mu_vecs[:,1,:])).T.dot(mu_vecs[:,0]))
    return mu_terms.sum() + kappa_terms.sum()    

def load_mixture_state(f):
    states_loaded = jnp.load(f)
    return MixtureFitState(**{f : jnp.array(states_loaded.get(f)) for f in states_loaded.files})
