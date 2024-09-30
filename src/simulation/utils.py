import jax
from jax import numpy as jnp, vmap
from jax.random import PRNGKey


def generate_mixture_samples(key: PRNGKey, credal_samples: jnp.ndarray, mixing_weights: jnp.ndarray):
    num_samples, num_extremes = credal_samples.shape[0], credal_samples.shape[2]
    sampled_mixture_classes = jax.random.choice(key, jnp.arange(num_extremes), shape=(num_samples,), p=mixing_weights)
    joint_idxs = jnp.stack([jnp.arange(num_samples), sampled_mixture_classes], axis=1)

    def sample_from_class(args):
        sample_idx, class_idx = args
        return credal_samples[sample_idx, :, class_idx]

    return vmap(sample_from_class)(joint_idxs)


def runif_in_simplex(key: PRNGKey, num_extremes: int):
    ''' Return uniformly random vector in the simplex '''

    k = jax.random.exponential(key, shape=(num_extremes,), dtype=jnp.float32)
    return k / k.sum()


def initialise_moments_on_circle(num_extremes: int, radius: float = 3.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    angles = jnp.arange(0, 2 * jnp.pi, 2 * jnp.pi / num_extremes)
    means = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)
    cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    return means, cov


def initialise_moments(num_extremes: int, n_dim: int):
    means = jax.random.normal(jax.random.PRNGKey(0), shape=(num_extremes, n_dim)) * 3.0
    cov = jnp.eye(n_dim)
    return means, cov


import jax.numpy as jnp
from jax import random


def initialise_moments_on_sphere(num_extremes: int, radius: float = 3.0, dim: int = 3, key=random.PRNGKey(0)) -> tuple[
    jnp.ndarray, jnp.ndarray]:
    """
    Initializes means and covariance matrix for points on a d-dimensional sphere.

    Args:
    - num_extremes: The number of extreme points to generate on the sphere.
    - radius: The radius of the sphere.
    - dim: The dimensionality of the sphere.
    - key: A PRNG key for generating random points (for reproducibility).

    Returns:
    - means: A (num_extremes, dim) array containing the means (points) on the sphere.
    - cov: A (dim, dim) covariance matrix (identity for simplicity).
    """
    # Generate random points on the unit sphere
    key, subkey = random.split(key)
    points = random.normal(subkey, (num_extremes, dim))

    # Normalize to project the points onto the sphere surface
    points = points / jnp.linalg.norm(points, axis=1, keepdims=True)

    # Scale by the specified radius
    means = radius * points

    # Covariance matrix as an identity matrix in dim dimensions
    cov = jnp.eye(dim)

    return means, cov
