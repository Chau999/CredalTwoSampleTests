import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey
from jax.scipy.stats import norm

from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.kernel_credal_discrepancy import compute_infMMDsq_batch
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmdq import compute_mmdq


def mmdq_test(key: PRNGKey,
              xs: Array,
              credal_ys: Array,
              kernel: Kernel,
              level: float = 0.05,
              return_p_val: bool = False
              ):
    # estimation
    credal_xs = jnp.stack([xs] * credal_ys.shape[2], axis=2)
    _, estimated_weights, errors = compute_infMMDsq_batch(credal_ys, credal_xs, kernel)
    if (estimated_weights.sum() > 1.05) or (estimated_weights.sum() <= 0):
        return None

    psuedo_ys = generate_mixture_samples(key, credal_ys, estimated_weights)
    test_statistic = compute_mmdq(xs, psuedo_ys, kernel)

    if return_p_val:
        return norm.sf(test_statistic)
    else:
        critical_value = norm.ppf(1 - level)
        return (test_statistic > critical_value).astype(jnp.int32)
