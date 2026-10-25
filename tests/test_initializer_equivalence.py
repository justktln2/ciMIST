"""Preserve the original k-means++ draws, masks and concentration keys exactly."""
from unittest import TestCase, main
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.ci import vmm

# Frozen from both PR baselines (0ea26a6 and 9f30db0). Keep the original
# recurrence here so a future optimization cannot redefine its own reference.
def original_spherical_kmeans_plus_plus(theta, n_clusters, key):
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
    ix = jax.random.randint(keys[0], (1,), 0, theta.shape[0])

    inv_norm = 1/jnp.linalg.norm(theta, axis=1)
    #theta_hat = jnp.diag(1/norm) @ theta # SAMPLES x 28, THIS CAUSES MEMORY PROBLEMS
    theta_hat = theta * inv_norm[:, jnp.newaxis]

    normalized_cluster_centers = theta_hat[ix].T
    ixs = [ix]

    for k in keys[1:]:
        # compute cosine distances to cluster centers
        cosine_distances = 1 - theta_hat @ normalized_cluster_centers

        # probability of choosing cluster center is proportional to min_distance**2
        min_distance = jnp.min(cosine_distances, axis=1)
        log_p = jnp.where(min_distance > 1e-8,
                          2*jnp.log(min_distance),
                          -jnp.inf # probability of zero for something already chosen
                         ) # (un-normalized) log probability of choosing index
        ix = jax.random.categorical(k, log_p, shape=(1,))

        new_center = theta_hat[ix].T

        normalized_cluster_centers = jnp.hstack((normalized_cluster_centers, new_center)).squeeze()
        ixs.append(ix)

    ixs = jnp.concatenate(ixs)
    centers = theta[ixs]
    return centers


class InitializerEquivalenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(197)
        points = rng.uniform(-2.5, 2.5, (17, 28))
        points[:, ::5] = 0
        cls.points = points
        cls.cases = {
            "random": (points, 7), "one_component": (points, 1),
            "duplicates": (np.repeat(points[:3], 3, axis=0), 12),
            "constant": (np.tile(points[0], (5, 1)), 9),
            "zero": (np.zeros((5, 28)), 9),
            "one_frame": (points[:1], 7), "two_frames": (points[:2], 7),
        }
        base = np.linspace(0.5, 2.5, 28)
        for scale in (1e-3, 1e-4, 1e-5):
            cls.cases[f"nearly_constant_{scale}"] = (
                base + scale * rng.normal(size=(17, 28)), 7)

    def test_center_draws_match_original(self):
        for name, (points, components) in self.cases.items():
            for seed in (0, 1, 7, 42):
                with self.subTest(case=name, seed=seed):
                    theta, key = jnp.asarray(points), jax.random.PRNGKey(seed)
                    expected = original_spherical_kmeans_plus_plus(theta, components, key)
                    actual = vmm.spherical_kmeans_plus_plus(theta, components, key)
                    np.testing.assert_array_equal(actual, expected)

    def test_initialized_parameters_and_keys_match_original(self):
        theta = jnp.asarray(self.points)
        mask = jnp.asarray(np.arange(28) % 5 != 0)
        for seed in (0, 1, 7, 42):
            with self.subTest(seed=seed):
                key = jax.random.PRNGKey(seed)
                with patch.object(vmm, "spherical_kmeans_plus_plus", original_spherical_kmeans_plus_plus):
                    expected = vmm.init_random_mixture_state(theta, mask, key, n_components=100)
                actual = vmm.init_random_mixture_state(theta, mask, key, n_components=100)
                # Responsibilities and history slots are intentionally unwritten.
                for name in ("mu", "kappa", "logw", "mask", "n_components", "n_iter", "converged", "atol"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)

    def test_batched_compiled_centers_match_original(self):
        theta = jnp.asarray(self.points)
        keys = jnp.stack([jax.random.PRNGKey(seed) for seed in (0, 1, 7, 42)])
        expected = jax.vmap(lambda key: original_spherical_kmeans_plus_plus(theta, 100, key))(keys)
        actual = jax.jit(jax.vmap(lambda key: vmm.spherical_kmeans_plus_plus(theta, 100, key)))(keys)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    main()
