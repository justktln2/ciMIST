"""Check normalized fixed-responsibility objective/gradient, including zero weights."""
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import i0, xlogy
import numpy as np
from cimist.ci.vmm import expected_log_likelihood, mu_mle

class StatisticsTests(unittest.TestCase):
    def test_objective_and_gradient_match_frame_expression(self):
        rng = np.random.default_rng(61)
        theta = jnp.asarray(rng.uniform(-np.pi, np.pi, (41, 3)))
        r = rng.uniform(0, 1, (41, 4))
        r[:, -1] = 0
        r /= r.sum(axis=1, keepdims=True)
        r = jnp.asarray(r)
        w = r.sum(axis=0)
        mu = mu_mle(theta, r)
        mask = jnp.array([True, False, True])
        c = (r.T @ jnp.cos(theta)) * jnp.cos(mu) + (r.T @ jnp.sin(theta)) * jnp.sin(mu)
        def clip(p):
            return jnp.where(mask, jnp.clip(p, 10., 600.), 0.)
        def original(p):
            return -expected_log_likelihood(theta, mu, clip(p), r, w, mask) / len(theta)
        def statistics(p):
            k = clip(p)
            terms = jnp.where(mask, k*c - w[:, None]*jnp.log(2*jnp.pi*i0(k)), 0.)
            return -(terms.sum() + xlogy(w,w).sum()) / len(theta)
        for value in (5., 10., 47., 600., 650.):
            p = jnp.full((4,3), value)
            for reference, candidate in zip(jax.value_and_grad(original)(p), jax.value_and_grad(statistics)(p)):
                np.testing.assert_allclose(candidate, reference, rtol=1e-8, atol=1e-10)

if __name__ == "__main__":
    unittest.main()
