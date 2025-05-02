"""
Functions for entropy estimation
"""

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, polygamma, entr
from jax.scipy.linalg import eigh
from jax.nn import relu
from jax import Array, jit
from jax.typing import ArrayLike

# consistency with numpy
loggamma = gammaln


@jit
def S_vn(rho: Array) -> ArrayLike:
    """
    Compute the von Neumman entropy of the density overlap matrix rho.

    With a square matrix $\rho$ defined by $\rho_{ij} = \sqrt{p(i)}\sqrt{p(j)}$,
    the von Neumann entropy is given by $S_{vn} = -\sum_{j} \lambda_j \log \lambda_j$.
    """
    lambda_, _ = eigh(rho)
    return relu(entr(lambda_)).sum()


def S_mle(n: Array) -> ArrayLike:
    """
    Compute the maximum likelihood or "plugin" estimator of the entropy from a
    vector of counts.

    The maximum likelihood estimator is given by

    $$
    \hat{S}_{MLE} = -\sum_{i} \frac{n_i}\frac{N} \log \frac{n_i}\frac{N}.
    $$
    """
    n = jnp.array(n).flatten()
    p = n / n.sum()
    return entr(p).sum()


### Core functions on which Bayesian estimators all depend.
def S_posterior_mean(alpha: Array) -> ArrayLike:
    """
    Calculate expected entropy of a categorical distribution $p \sim Dirichlet(\alpha)$.

    References
    ----------
    [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research. 15(81):2833−2868, 2014.
        doi:10.5555/2627435.2697056

    Notes
    -----
    Equation 18 of appendix A.1 of ref [1].
    """
    Alpha = jnp.sum(alpha)
    return digamma(Alpha + 1) - jnp.sum(alpha * digamma(alpha + 1)) / Alpha


def S_posterior_mean_sq(alpha: Array) -> ArrayLike:
    """
    Calculate expected squared entropy of a categorical distribution $p \sim Dirichlet(\alpha)$.


    References
    ----------
    [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research. 15(81):2833−2868, 2014.
        doi:10.5555/2627435.2697056

    Notes
    -----
    See equation 19 in appendix A.2 of ref [1].
    """
    # "normalizing" constant
    A = jnp.sum(alpha)

    # compute the "diagonal" terms
    J = (
        (digamma(alpha + 2) - digamma(A + 2)) ** 2
        + polygamma(1, alpha + 2)
        - polygamma(1, A + 2)
    )
    diag_sum = jnp.dot(alpha * (alpha + 1), J)
    # "off-diagonal" terms
    partial_term = digamma(alpha + 1) - digamma(A + 2)
    off_diag_terms = jnp.outer(partial_term, partial_term) - polygamma(1, A + 2)
    alpha_outer = jnp.outer(alpha, alpha)
    off_diag_sum = jnp.sum(off_diag_terms * alpha_outer) - jnp.sum(
        jnp.diag(off_diag_terms * alpha_outer)
    )
    return (diag_sum + off_diag_sum) / (A * (A + 1))


def S_posterior_moments(alpha: Array) -> Array:
    """
    References
    ----------
    [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research. 15(81):2833−2868, 2014.
        doi:10.5555/2627435.2697056

    Notes
    -----
    See equations 18-19 in appendix A of reference [1].
    """
    alpha = alpha.flatten()
    return jnp.array([S_posterior_mean(alpha), S_posterior_mean_sq(alpha)])


@jit
def S_posterior_mean_std(alpha: Array) -> Array:
    """
    Calculate the mean and standard deviation of the distribution of the entropy of a categorical
    distribution $p \sim Dirichlet(\alpha)$.

    Returns
    -------
    A 2x1 array with the posterior mean as the first entropy and the posterior standard deviation as the second.

    References
    ----------
    [1] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research. 15(81):2833−2868, 2014.
        doi:10.5555/2627435.2697056

    Notes
    -----
    See equations 18-19 in appendix A of reference [1].
    """
    S_mean, S_mean_sq = S_posterior_moments(alpha)
    return jnp.array([S_mean, jnp.sqrt(S_mean_sq - S_mean**2)])


### Dirichlet hyperprior-based estimator
def nsb(n):
    """
    Calculate the Nemenman-Shafee-Bialek estimator of the entropy from a vector of counts.


    References
    ----------
    [1] Ilya Nemenman, Fariel Shafee, and William Bialek; NeurIPS 2001.
        doi:10.5555/2980539.2980601
    [2] Evan Archer, Il Memming Park, Jonathan W. Pillow; Journal of Machine Learning Research. 15(81):2833−2868, 2014.
        doi:10.5555/2627435.2697056
    [3] Damián G Hernández,Ahmed Roman, Ilya Nemenman; Phys Rev E. 2023 Jul;108(1-1):014101.
        doi:10.1103/PhysRevE.108.014101

    Notes
    -----
    See:
        - 5-9 in reference [3]
        - 18-19 in appendix A of reference [2]
    """
    A = len(n)
    N = jnp.sum(n)

    def logfactorial(n):
        return loggamma(n + 1)

    # Make functions to integrate
    def log_p(n, lambda_):
        # log of equation 9 from HRN
        # log is used for the sake of numerical
        # stability.
        # We can probably evaluate this by
        # saddle point anyway
        return (
            logfactorial(N)
            + loggamma(A * lambda_)
            + A * loggamma(N + A * lambda_)
            + jnp.sum(loggamma(n + lambda_) - logfactorial(n))
        )

    def p_prior(lambda_):
        return A * polygamma(1, A * lambda_ + 1) - polygamma(1, lambda_ + 1)

    def S(n, lambda_):
        S_posterior_mean(n + lambda_ + 1)

    raise NotImplementedError


def I_hs(
    n_xy,
    beta,
):
    """
    Estimate mutual information between X and Y using the Hernandez-Samengo estimator at fixed beta.


    References
    ----------
    [1] DG Hernández, I Samengo. Entropy 21 (6), 623, 2019. 18, 2019.
    [2] Damián G Hernández, Ahmed Roman, Ilya Nemenman; Phys Rev E. 2023 Jul;108(1-1):014101.
        doi:10.1103/PhysRevE.108.014101

    Notes
    -----
    See:
        - Equation 16 in reference [1]
    """

    n_x = n_xy.sum(axis=1)
    p_x = n_x / n_x.sum()
    n_y = n_xy.sum(axis=0)
    p_y = n_y / n_y.sum()

    H_y = S_mle(p_y)

    I_hat = H_y - jnp.sum(
        p_x
        * (
            digamma(beta + n_x + 1)
            - jnp.sum(
                ((beta * p_y + n_xy) / jnp.expand_dims(beta + n_x, 1))
                * digamma(beta * p_y + n_xy + 1),
                axis=1,
            )
        )
    )
    return I_hat


def dirichlet_marginal_log_likelihood(beta, n_xy, p_beta=1, p_n=1):
    """
    Estimate the marginal likeilihood of n_xy under parameter beta.
    Used for constructing the full Hernandez-Samengo estimator.


    References
    ----------
    [1] DG Hernández, I Samengo. Entropy 21 (6), 623, 2019. 18, 2019.

    Notes
    -----
    See:
        - Equation 14 in reference [1]
    """

    N = n_xy.sum()  # TODO: why is this here?  # noqa: F841
    X, Y = n_xy.shape
    if Y > X:
        n_xy = n_xy.T
    n_x = n_xy.sum(axis=1)
    p_x = n_x / n_x.sum()  # TODO: why is this here? # noqa: F841
    n_y = n_xy.sum(axis=0)
    p_y = n_y / n_y.sum()

    # cardinality of set X
    _X_ = jnp.sum(n_x != 0)  # TODO: why is this here? # noqa: F841
    _Y_ = jnp.sum(n_y != 0)  # TODO: why is this here? # noqa: F841

    return (
        jnp.log(p_beta)
        - jnp.log(p_n)
        + jnp.sum(
            (loggamma(beta) - loggamma(n_x + beta))
            + jnp.sum(
                loggamma(n_xy + beta * p_y) - jnp.sum(loggamma(beta * p_y), axis=1)
            )
        )
    )


def S_block_jackknife(state_traj, ix):
    states, counts = jnp.unique(state_traj, return_counts=True)
    S_resamp = []
    counts_resamp = []  # TODO: not used # noqa: F841
    lifetimes = []
    for s in states:
        state_indicator = jnp.diff(state_traj == s)
        mean_lifetime = jnp.mean(jnp.diff(jnp.where(state_indicator != 0)[0]))
        lifetimes.append(mean_lifetime)
        counts_ = counts.at[s].set(counts[s] - mean_lifetime)
        counts_ = jnp.where(counts_ > 0, counts_, 1)
        S_resamp.append(S_posterior_moments(jnp.asarray(counts_)))
        # counts_ = counts.at[s].set(counts[s] + mean_lifetime)
        # counts_ = jnp.where(counts_ > 0, counts_, 0)
        # S_resamp.append(ar.entropy_estimators.S_posterior_moments(counts_))
        # counts_resamp.append(counts_)
    S_resamp = jnp.array(S_resamp)
    mu = jnp.mean(S_resamp[:, 0])
    mu_sq = jnp.mean(S_resamp[:, 1])
    se = jnp.sqrt(mu_sq - mu**2)
    return se
