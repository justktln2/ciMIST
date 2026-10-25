"""Post-M-step likelihoods must describe the parameters accepted by EM.

Paper S3, lines 841--843: stop at 0.01 nats per observation. Independent
SciPy evaluations use rtol=1e-12, atol=1e-12 on fixed float64 inputs.
"""
import unittest
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import i0e, logsumexp

from cimist.ci import vmm


def independent_log_likelihood(theta, state):
    theta, mu, kappa, logw, mask = map(
        np.asarray, (theta, state.mu, state.kappa, state.logw, state.mask))
    components = np.where(
        mask, kappa[:, None, :] * np.cos(theta[None, :, :] - mu[:, None, :])
        - np.log(2 * np.pi) - np.log(i0e(kappa[:, None, :]))
        - np.abs(kappa[:, None, :]), 0).sum(axis=-1)
    return logsumexp(components + (logw - logsumexp(logw))[:, None], axis=0).mean()


class EMConvergence(unittest.TestCase):
    def test_each_solver_scores_the_returned_m_step_parameters(self):
        rng = np.random.default_rng(20261024)
        theta = jnp.pad(jnp.asarray(rng.normal(
            [0.4, -0.2, 1.0], [.2, .25, .8], (97, 3))), ((0, 0), (0, 25)))
        initial = vmm.init_random_mixture_state(
            theta, jnp.arange(28) < 2, jax.random.PRNGKey(7),
            n_components=3, maxiter=100)
        initial_ll = independent_log_likelihood(theta, initial)
        for solver in ("newton", "bfgs"):
            with self.subTest(solver=solver):
                state = vmm.em_step(theta, initial, solver=solver)
                self.assertEqual(int(state.n_iter), 1)
                self.assertFalse(bool(state.converged))
                self.assertFalse(bool(vmm.is_converged(state)))
                self.assertGreater(abs(float(state.log_likelihood[0]) - initial_ll), .1)
                for iteration in (1, 2):
                    if iteration == 2:
                        state = vmm.em_step(theta, state, solver=solver)
                    np.testing.assert_allclose(
                        state.log_likelihood[iteration - 1],
                        independent_log_likelihood(theta, state), rtol=1e-12, atol=1e-12)
                self.assertEqual(bool(state.converged), bool(vmm.is_converged(state)))

    def test_small_previous_gain_cannot_hide_a_large_latest_gain(self):
        # The original lagged check accepts update 2: L1-L0 < .01, although
        # the returned parameters have L2-L1 > .01. No optimizer is involved.
        theta = jnp.full((9, 28), .5)
        state = vmm.init_random_mixture_state(
            theta, jnp.arange(28) == 0, jax.random.PRNGKey(0), n_components=1)
        state = state._replace(mu=jnp.zeros((1, 28)), kappa=10. * state.mask[None, :])
        ll0 = independent_log_likelihood(theta, state)
        result = vmm.ScalarSolveResult(True, 0, 1, 1, 1, 0)
        updates = [(state.mu.at[0, 0].set(mu), state.kappa, state.logw, result)
                   for mu in (.001, .5, .5)]
        with jax.disable_jit(), patch.object(vmm, "m_step", side_effect=updates):
            first = vmm.em_step(theta, state)
            second = vmm.em_step(theta, first)
            self.assertLess(abs(float(first.log_likelihood[0]) - ll0), float(state.atol))
            self.assertGreater(float(second.log_likelihood[1] - second.log_likelihood[0]),
                               float(state.atol))
            self.assertFalse(bool(second.converged))
            third = vmm.em_step(theta, second)
            self.assertTrue(bool(third.converged))
            self.assertEqual(int(third.n_iter), 3)

    def test_real_newton_fit_stops_on_the_latest_likelihood(self):
        rng = np.random.default_rng(19)
        labels = np.arange(240) % 2
        theta = jnp.pad(jnp.asarray(rng.normal(
            np.array([-.7, .8])[labels, None], .18, (240, 2))), ((0, 0), (0, 26)))
        initial = vmm.init_random_mixture_state(
            theta, jnp.arange(28) < 2, jax.random.PRNGKey(4), n_components=4)
        state = initial
        observed = []
        for _ in range(100):
            state = vmm.step_if_not_converged(theta, state)
            observed.append(independent_log_likelihood(theta, state))
            self.assertTrue(bool(state.successes[int(state.n_iter) - 1]))
            if bool(state.converged):
                break
        self.assertTrue(bool(state.converged))
        self.assertGreaterEqual(int(state.n_iter), 2)
        np.testing.assert_allclose(state.log_likelihood[:int(state.n_iter)], observed,
                                   rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(np.diff(observed) >= -1e-12))
        self.assertTrue(np.all(np.abs(np.diff(observed)[:-1]) >= float(state.atol)))
        self.assertLess(abs(observed[-1] - observed[-2]), float(state.atol))
        stopped = vmm.step_if_not_converged(theta, state)
        for actual, expected in zip(stopped, state):
            np.testing.assert_array_equal(actual, expected)
        refreshed = vmm.update_r(theta, state)
        np.testing.assert_allclose(refreshed.r, vmm.predict_proba(theta, refreshed),
                                   rtol=1e-12, atol=1e-12)
        # The multi-step API must stop at the same accepted update.
        advanced = vmm.advance_em(theta, initial, steps=100)
        expected = vmm.sort_by_weight(state)
        for field in ("mu", "kappa", "logw", "n_iter", "converged"):
            np.testing.assert_allclose(getattr(advanced, field), getattr(expected, field),
                                       rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
