import jax
import jax.numpy as jnp
from jax import vmap, Array
from jax.random import PRNGKey

from src.testing_algorithms.kernel_credal_discrepancy import compute_infinfMMDsq
from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmd import compute_MMDsq, compute_h_gram
from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.utils import shuffle_cut_and_compute_mmd, split_samples, compute_wild_bootstrap_samples


def credal_plausibility_test(key: PRNGKey,
                             credal_xs: Array,
                             credal_ys: Array,
                             kernel: Kernel,
                             split_ratio: float = 0.5,
                             num_permutations: int = 500,
                             permutation_type: str = "wild_bootstrap",
                             level: float = 0.05
                             ):
    # stage 1: determine train test split
    credal_xs_train, credal_xs_test, credal_ys_train, credal_ys_test = split_samples(credal_xs, credal_ys, split_ratio)

    # stage 2: compute mixing weight (using train data)
    weight_xs, weight_ys, _ = compute_infinfMMDsq(credal_xs_train, credal_ys_train, kernel)
    if weight_ys[-1].sum() > 1.05 or weight_ys[-1].sum() <= 0:
        return None
    if weight_xs[-1].sum() > 1.05 or weight_xs[-1].sum() <= 0:
        return None

    # stage 3: compute critical value (using test data)
    psuedo_xs = generate_mixture_samples(key, credal_xs_test, weight_xs[-1])
    psuedo_ys = generate_mixture_samples(key, credal_ys_test, weight_ys[-1])
    observed_mmd = compute_MMDsq(psuedo_xs, psuedo_ys, kernel)

    # simulate the null
    if permutation_type == "permutation_test":
        joint_samples = jnp.concat([psuedo_xs, psuedo_ys])
        keys = jax.random.split(key, num_permutations)
        permutations = vmap(lambda key: jax.random.permutation(key, joint_samples.shape[0]))(keys)
        simulated_mmds = vmap(lambda permutation: shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel))(
            permutations)
        simulated_mmds = jnp.append(simulated_mmds, observed_mmd)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    elif permutation_type == "wild_bootstrap":
        h_gram = compute_h_gram(psuedo_xs, psuedo_ys, kernel)
        simulated_mmds = compute_wild_bootstrap_samples(key, h_gram, num_permutations)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    else:
        raise ValueError(f"Permutation type {permutation_type} not found.")

    return (observed_mmd > critical_value).astype(jnp.int32)
