import jax
import jax.numpy as jnp
from jax import vmap, Array
from jax.random import PRNGKey

from src.testing_algorithms.kernel_credal_discrepancy import compute_infMMDsq_batch,compute_infMMDsq
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmd import compute_MMDsq, compute_h_gram
from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.utils import shuffle_cut_and_compute_mmd, split_samples, compute_wild_bootstrap_samples


def credal_specification_test(key: PRNGKey,
                              xs: Array,
                              credal_ys: Array,
                              kernel: Kernel,
                              split_ratio: float = 0.5,
                              num_permutations: int = 500,
                              permutation_type: str = "wild_bootstrap",
                              level: float = 0.05,
                              return_p_val: bool = False,
                              robustness: float = 0.0
                              ):
    # stage 1: determine train test split
    xs_train, xs_test, credal_ys_train, credal_ys_test = split_samples(xs, credal_ys, split_ratio)

    # stage 2: compute mixing weight (using train data)
    credal_xs_train = jnp.stack([xs_train] * credal_ys_train.shape[2], axis=2)
    observed_inf_mmd, estimated_weights, error = compute_infMMDsq_batch(credal_ys_train, credal_xs_train, kernel)
    if (estimated_weights.sum() > 1.05) or (estimated_weights.sum() <= 0):
            return None

    observed_inf_mmd, estimated_weights = compute_infMMDsq(credal_xs=credal_ys_train, ys=xs_train, kernel=kernel)

    # stage 3: compute critical value (using test data)
    psuedo_ys_test = generate_mixture_samples(key, credal_ys_test, estimated_weights)
    observed_mmd = compute_MMDsq(psuedo_ys_test, xs_test, kernel)

    # print("estimated_weights", estimated_weights)

    # simulate the null
    if permutation_type == "permutation_test":
        joint_samples = jnp.concat([xs_test, psuedo_ys_test])
        keys = jax.random.split(key, num_permutations)
        permutations = vmap(lambda key: jax.random.permutation(key, joint_samples.shape[0]))(keys)
        simulated_mmds = vmap(lambda permutation: shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel))(
            permutations)
        simulated_mmds = jnp.append(simulated_mmds, observed_mmd)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    elif permutation_type == "wild_bootstrap":
        h_gram = compute_h_gram(xs_test, psuedo_ys_test, kernel)
        simulated_mmds = compute_wild_bootstrap_samples(key, h_gram, num_permutations)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    else:
        raise ValueError(f"Permutation type {permutation_type} not found.")

    if return_p_val:
        return (simulated_mmds >= observed_mmd).mean()
    else:
        return (observed_mmd > critical_value + (2*robustness*jnp.sqrt(2)/psuedo_ys_test.shape[0])).astype(jnp.int32)
        # return (observed_mmd > critical_value ).astype(jnp.int32)

    # return xs_train.shape[0] * observed_inf_mmd, estimated_weights, error
