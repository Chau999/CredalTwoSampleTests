from jax import Array, numpy as jnp
from jax.random import PRNGKey
from jax.scipy.stats import norm

from src.testing_algorithms.kernel_credal_discrepancy import compute_infinfMMDsq
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmdq import compute_mmdq_star
from src.simulation.utils import generate_mixture_samples


def mmdq_star_intersection_test(key: PRNGKey,
                                credal_xs: Array,
                                credal_ys: Array,
                                kernel: Kernel,
                                eps_power: float = -1 / 20,
                                level: float = 0.05,
                                weight_x: Array = None,
                                weight_y: Array = None
                                ):
    # estimate weights
    if (weight_x is None) and (weight_y is None):
        weight_xs, weight_ys, _ = compute_infinfMMDsq(credal_xs, credal_ys, kernel)
        if weight_ys[-1].sum() > 1.05 or weight_ys[-1].sum() <= 0:
            return None
        if weight_xs[-1].sum() > 1.05 or weight_xs[-1].sum() <= 0:
            return None
        weight_x, weight_y = weight_xs[-1], weight_ys[-1]

    psuedo_xs = generate_mixture_samples(key, credal_xs, weight_x)
    psuedo_ys = generate_mixture_samples(key, credal_ys, weight_y)

    test_statistic = compute_mmdq_star(psuedo_xs, psuedo_ys, kernel, eps_power)
    critical_value = norm.ppf(1 - level)

    return (test_statistic > critical_value).astype(jnp.int32)
