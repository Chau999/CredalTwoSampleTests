import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from src.simulation.utils import generate_mixture_samples, runif_in_simplex, initialise_moments


def generate_corner_student_samples(key: PRNGKey, means: jnp.ndarray, num_samples: int, df: int = 1):
    shifted_samples = []
    keys = jax.random.split(key, means.shape[0])
    for i, mean in enumerate(means):
        sample = jax.random.t(keys[i], df=df, shape=(num_samples, means.shape[1]))
        shifted_sample = sample + mean
        shifted_samples.append(shifted_sample)
    return jnp.stack(shifted_samples, axis=2)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    num_extremes = 5
    num_samples = 1000
    mixing_weights = runif_in_simplex(key1, num_extremes)
    means, cov = initialise_moments(num_extremes, 2)
    means = jax.numpy.abs(means)
    samples_from_corners = generate_corner_student_samples(key2, means, num_samples)
    samples = generate_mixture_samples(key2, samples_from_corners, mixing_weights)
    print(samples.shape)
    print("Done!")
