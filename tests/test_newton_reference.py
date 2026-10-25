"""Independent bracketed reference for the bounded Newton M-step."""
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e
from cimist.ci.vmm import m_step, mu_mle

class NewtonReferenceTests(unittest.TestCase):
    def test_bounds_masks_zero_weights_and_interior_roots(self):
        rng = np.random.default_rng(17)
        theta = jnp.asarray(np.column_stack([rng.normal(0, .15, 251), rng.uniform(-np.pi,np.pi,251), rng.normal(0,.01,251)]))
        r = jnp.asarray(np.column_stack([np.ones(251), np.zeros(251)]))
        initial = jnp.full((2,3), 30.)
        for mask in (jnp.array([True, True, True]), jnp.array([True, False, True])):
            mu, kappa, _, result = m_step(theta, jnp.zeros_like(initial), initial, r, mask, solver="newton")
            self.assertTrue(bool(result.success))
            mean = mu_mle(theta,r)
            c = np.asarray((r.T @ jnp.cos(theta))*jnp.cos(mean) + (r.T @ jnp.sin(theta))*jnp.sin(mean))
            expected = np.full((2,3),30.)
            for j in range(3):
                ratio = c[0,j] / 251
                f = lambda x: i1e(x)/i0e(x)-ratio
                expected[0,j] = 10. if f(10.) >= 0 else 600. if f(600.) <= 0 else brentq(f,10.,600.,xtol=1e-12)
            expected[:,~np.asarray(mask)] = 0.
            np.testing.assert_allclose(kappa,expected,rtol=1e-8,atol=1e-10)

if __name__ == "__main__":
    unittest.main()
