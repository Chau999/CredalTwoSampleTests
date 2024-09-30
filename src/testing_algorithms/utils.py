import jax
from jax import vmap

from src.testing_algorithms.mmd import compute_MMDsq


def compute_p_value(simulated_statistics: jax.Array, observed_statistic: jax.Array) -> jax.Array:
    return (simulated_statistics >= observed_statistic).mean()


def compute_root_split_ratio(num_samples, power=0.5, tolerance=1e-3, max_iterations=1000):
    """This is to determine the adaptive splitting ratio for the two-stage approaches."""

    def f(m):
        return m + m ** power - num_samples

    def f_prime(m):
        return 1 + power * m ** (power - 1)

    m = num_samples / 2  # Initial guess
    for _ in range(max_iterations):
        m_new = m - f(m) / f_prime(m)
        if abs(m_new - m) < tolerance:
            return m_new / num_samples
        m = m_new

    raise ValueError("Solution did not converge")


def shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel):
    num_samples = int(joint_samples.shape[0] / 2)
    left_samples, right_samples = joint_samples[permutation][: num_samples], joint_samples[permutation][num_samples:]
    return compute_MMDsq(left_samples, right_samples, kernel)


def split_samples(samples, samples_from_corners, split_ratio):
    n = int(samples_from_corners.shape[0] * split_ratio)
    samples_train, samples_test = samples[:n], samples[n:]  # use less sample to test.
    samples_from_corners_train, samples_from_corners_test = samples_from_corners[:n], samples_from_corners[n:]
    return samples_train, samples_test, samples_from_corners_train, samples_from_corners_test


def compute_wild_bootstrap_samples(key: jax.random.PRNGKey,
                                   h_gram: jax.Array,
                                   num_permutations: int) -> jax.Array:
    n = h_gram.shape[0]
    key, keys = jax.random.split(key)
    Ws = jax.random.rademacher(keys, (num_permutations, n))
    return vmap(lambda W: (W @ h_gram @ W.T) / n ** 2)(Ws)
