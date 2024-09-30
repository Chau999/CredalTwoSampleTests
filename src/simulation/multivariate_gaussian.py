import jax
from jax import vmap, numpy as jnp
from jax.random import PRNGKey

from src.simulation.utils import generate_mixture_samples, runif_in_simplex, initialise_moments


def generate_corner_gaussian_samples(key: PRNGKey, means: jnp.ndarray, cov: jnp.ndarray, num_samples: int):
    samples = vmap(lambda mean: jax.random.multivariate_normal(key, mean, cov, shape=(num_samples,)))(means)
    return jnp.stack(samples, axis=2)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    num_extremes = 5
    num_samples = 1000
    mixing_weights = runif_in_simplex(key1, num_extremes)
    means, cov = initialise_moments(num_extremes)
    samples_from_corners = generate_corner_gaussian_samples(key2, means, cov, num_samples)
    samples = generate_mixture_samples(key2, samples_from_corners, mixing_weights)
    print(samples.shape)
    print("Done!")
