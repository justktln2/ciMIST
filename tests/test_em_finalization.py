"""The production pipeline must finalize successful EM fits before clustering."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from cimist import fit
from cimist.ci import vmm


class ReachedClustering(Exception):
    pass


class EMFinalization(unittest.TestCase):
    def run_pipeline(self, final_update, solver="newton", failed_step=False,
                     preconverged=False):
        """Control convergence timing while exercising main's actual stage order."""
        rng = np.random.default_rng(20261022)
        angles = jnp.asarray(rng.uniform(-np.pi, np.pi, (2, 7, 2)))
        diagnostics = np.zeros((2, 100))
        initial = vmm.MixtureFitState(
            np.full(2, 3), rng.uniform(-np.pi, np.pi, (2, 3, 2)),
            np.full((2, 3, 2), 2.), np.full((2, 3), -np.log(3)),
            np.ones((2, 2), dtype=bool), np.full((2, 7, 3), 1 / 3),
            diagnostics, np.zeros(2, dtype=int), np.full(2, preconverged),
            np.full(2, 1e-2), diagnostics, diagnostics, diagnostics,
            diagnostics, diagnostics)
        last_state = initial

        def advance(theta, state):
            nonlocal last_state
            iterations = np.where(state.converged, state.n_iter, state.n_iter + 1)
            successes = state.successes.copy()
            statuses = state.statuses.copy()
            successes[np.arange(2), iterations - 1] = not failed_step
            statuses[np.arange(2), iterations - 1] = 1 if failed_step else 0
            last_state = state._replace(
                n_iter=iterations, converged=iterations >= [2, final_update],
                mu=initial.mu + iterations[:, None, None] / 100,
                successes=successes, statuses=statuses)
            # EM leaves responsibilities from before its last M-step.
            return last_state

        advance_spy = Mock(side_effect=advance)
        refresh_spy = Mock(wraps=jax.vmap(vmm.update_r))

        def map_stage(func, **kwargs):
            self.assertIs(func.func, vmm.step_if_not_converged)
            self.assertEqual(func.keywords,
                             dict(gtol=1e-3, gmaxiter=500, solver=solver))
            return advance_spy

        def map_residues(func):
            if func is vmm.update_r:
                return refresh_spy
            self.assertIs(func, vmm.init_random_mixture_state)
            return Mock(return_value=initial)

        clustering = Mock(side_effect=ReachedClustering)
        data = SimpleNamespace(
            traj=Mock(), angles=angles, cloud_mask=initial.mask,
            residues=["GLY1_A", "GLY2_A"])
        data.traj.__len__ = Mock(return_value=7)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            args = fit.parse_args(["-o", str(Path(folder) / "run"),
                                   "--solver", solver, "--VMM_BATCH_SIZE", "1"])
            stack.enter_context(patch.object(fit, "parse_args", return_value=args))
            stack.enter_context(patch.object(fit, "load_dataset", return_value=data))
            stack.enter_context(patch.object(fit.md, "load", return_value=Mock()))
            stack.enter_context(patch.object(fit, "vmap", side_effect=map_residues))
            stack.enter_context(patch.object(fit, "combine_mixture_fit_batches", return_value=initial))
            stack.enter_context(patch.object(fit, "jit", side_effect=lambda func: func))
            stack.enter_context(patch.object(fit, "cvmap", side_effect=map_stage))
            stack.enter_context(patch.object(fit.cst.ci.dbscan, "compute_distances", clustering))
            stack.enter_context(patch.object(fit.logging, "basicConfig"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if failed_step and solver == "newton":
                error = "Newton M-step failed for residue indices"
            elif final_update > 100 and not preconverged:
                error = r"EM did not converge after 100 updates for residue indices \[1\]"
            else:
                error = None
            if error is not None:
                with self.assertRaisesRegex(RuntimeError, error):
                    fit.main()
                clustering.assert_not_called()
                refresh_spy.assert_not_called()
            else:
                with self.assertRaises(ReachedClustering):
                    fit.main()
                refresh_spy.assert_called_once()
                clustering.assert_called_once()
                final_state = clustering.call_args.args[0]
                expected = jax.vmap(vmm.update_r)(angles, last_state)
                for actual_field, expected_field in zip(final_state, expected):
                    np.testing.assert_allclose(actual_field, expected_field,
                                               rtol=1e-12, atol=1e-12)
                self.assertFalse(np.array_equal(final_state.r, last_state.r))
                np.testing.assert_allclose(final_state.r.sum(axis=-1), 1,
                                           rtol=1e-12, atol=1e-12)
            self.assertEqual(list(Path(folder).rglob("*.h5")), [])
            self.assertEqual(list(Path(folder).rglob("*.pkl")), [])
        return advance_spy.call_count

    def test_convergence_on_update_99_refreshes_once(self):
        for solver in ("newton", "bfgs"):
            with self.subTest(solver=solver):
                self.assertEqual(self.run_pipeline(99, solver), 99)

    def test_convergence_on_update_100_refreshes_once(self):
        for solver in ("newton", "bfgs"):
            with self.subTest(solver=solver):
                self.assertEqual(self.run_pipeline(100, solver), 100)

    def test_unfinished_update_100_never_refreshes_or_exports(self):
        for solver in ("newton", "bfgs"):
            with self.subTest(solver=solver):
                self.assertEqual(self.run_pipeline(101, solver), 100)

    def test_preconverged_state_refreshes_without_an_m_step(self):
        self.assertEqual(self.run_pipeline(0, preconverged=True), 0)

    def test_newton_failure_still_stops_before_clustering(self):
        self.assertEqual(self.run_pipeline(2, failed_step=True), 1)

    def test_bfgs_failure_diagnostics_do_not_change_legacy_behavior(self):
        self.assertEqual(self.run_pipeline(2, "bfgs", failed_step=True), 2)


if __name__ == "__main__":
    unittest.main()
