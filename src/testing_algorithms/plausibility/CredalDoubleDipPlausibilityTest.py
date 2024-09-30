import jax
import jax.numpy as jnp
from jax import vmap, Array
from jax.random import PRNGKey

from src.testing_algorithms.kernel_credal_discrepancy import compute_infinfMMDsq
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmd import compute_MMDsq, compute_h_gram
from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.utils import shuffle_cut_and_compute_mmd, compute_wild_bootstrap_samples


def two_staged_double_dip_split_inf_inf_mmd_test(key: PRNGKey,
                                                 credal_xs: Array,
                                                 credal_ys: Array,
                                                 kernel: Kernel,
                                                 split_power: float = 0.5,
                                                 num_permutations: int = 300,
                                                 permutation_type: str = "wild_bootstrap",
                                                 level: float = 0.05,
                                                 weight_x: Array = None,
                                                 weight_y: Array = None
                                                 ):
    # step 1: compute mixing weight
    if (weight_x is None) and (weight_y is None):
        weight_xs, weight_ys, _ = compute_infinfMMDsq(credal_xs, credal_ys, kernel)
        if weight_ys[-1].sum() > 1.05 or weight_ys[-1].sum() <= 0:
            return None
        if weight_xs[-1].sum() > 1.05 or weight_xs[-1].sum() <= 0:
            return None
        weight_x, weight_y = weight_xs[-1], weight_ys[-1]

    # step 2: subsampling
    num_testing_samples = int(credal_xs.shape[0] ** split_power)
    key, subkey = jax.random.split(key)
    subsample_indices = jax.random.choice(subkey, credal_xs.shape[0], (num_testing_samples,), replace=False)
    sub_credal_xs, sub_credal_ys = credal_xs[subsample_indices], credal_ys[subsample_indices]

    # step 3: compute critical value
    psuedo_xs = generate_mixture_samples(key, sub_credal_xs, weight_x)
    psuedo_ys = generate_mixture_samples(key, sub_credal_ys, weight_y)
    observed_mmd = compute_MMDsq(psuedo_xs, psuedo_ys, kernel)

    if permutation_type == "permutation_test":
        joint_samples = jnp.concat([psuedo_xs, psuedo_ys])
        keys = jax.random.split(key, num_permutations)
        permutations = vmap(lambda key: jax.random.permutation(key, joint_samples.shape[0]))(keys)
        simulated_mmds = vmap(lambda permutation: shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel))(
            permutations)
        simulated_mmds = jnp.append(simulated_mmds, observed_mmd)
        critical_value = jnp.quantile(simulated_mmds, 1-level)
    elif permutation_type == "wild_bootstrap":
        h_gram = compute_h_gram(psuedo_xs, psuedo_ys, kernel)
        simulated_mmds = compute_wild_bootstrap_samples(key, h_gram, num_permutations)
        critical_value = jnp.quantile(simulated_mmds, 1-level)
    else:
        raise ValueError(f"Permutation type {permutation_type} not found.")

    return (observed_mmd > critical_value).astype(jnp.int32)
