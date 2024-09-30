import jax
import jax.numpy as jnp
from jax import vmap, Array
from jax.random import PRNGKey

from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.kernel_credal_discrepancy import compute_infMMDsq_batch
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmd import compute_MMDsq, compute_h_gram
from src.testing_algorithms.utils import shuffle_cut_and_compute_mmd, compute_wild_bootstrap_samples


def credal_double_dip_specification_test(key: PRNGKey,
                                         xs: Array,
                                         credal_ys: Array,
                                         kernel: Kernel,
                                         split_power: float = 0.5,
                                         num_permutations: int = 300,
                                         permutation_type: str = "wild_bootstrap",
                                         level: float = 0.05,
                                         return_p_val: bool = False,
                                         robustness: float = 0.05
                                         ):
    # stage 1: compute mixing weight
    credal_xs = jnp.stack([xs] * credal_ys.shape[2], axis=2)
    observed_inf_mmd, estimated_weights, _ = compute_infMMDsq_batch(credal_ys, credal_xs, kernel)
    if (estimated_weights.sum() > 1.05) or (estimated_weights.sum() <= 0):
        return None

    # stage 2: subsample
    num_testing_samples = int(xs.shape[0] ** split_power)
    key, subkey = jax.random.split(key)
    subsample_indices = jax.random.choice(subkey, xs.shape[0], (num_testing_samples,), replace=False)
    sub_xs, sub_credal_ys = xs[subsample_indices], credal_ys[subsample_indices]

    # stage 3: compute critical value
    psuedo_sub_ys = generate_mixture_samples(key, sub_credal_ys, estimated_weights)
    observed_mmd = compute_MMDsq(psuedo_sub_ys, sub_xs, kernel)

    # simulate the null
    if permutation_type == "permutation_test":
        joint_samples = jnp.concat([sub_xs, psuedo_sub_ys])
        keys = jax.random.split(key, num_permutations)
        permutations = vmap(lambda key: jax.random.permutation(key, joint_samples.shape[0]))(keys)
        simulated_mmds = vmap(lambda permutation: shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel))(
            permutations)
        simulated_mmds = jnp.append(simulated_mmds, observed_mmd)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    elif permutation_type == "wild_bootstrap":
        h_gram = compute_h_gram(sub_xs, psuedo_sub_ys, kernel)
        simulated_mmds = compute_wild_bootstrap_samples(key, h_gram, num_permutations)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    else:
        raise ValueError(f"Permutation type {permutation_type} not found.")

    if return_p_val:
        return (simulated_mmds >= observed_mmd).mean()
    else:
        return (observed_mmd > critical_value + 2 * robustness * jnp.sqrt(2) / psuedo_sub_ys.shape[0]).astype(jnp.int32)
