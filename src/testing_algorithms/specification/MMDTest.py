import jax
import jax.numpy as jnp
from jax import vmap, Array
from jax.random import PRNGKey

from src.testing_algorithms.kernels import Kernel
from src.testing_algorithms.mmd import compute_MMDsq, compute_h_gram
from src.simulation.utils import generate_mixture_samples
from src.testing_algorithms.utils import shuffle_cut_and_compute_mmd, compute_wild_bootstrap_samples


def mmd_test(key: PRNGKey,
             xs: Array,
             credal_ys: Array,
             kernel: Kernel,
             num_permutations: int = 300,
             permutation_type: str = "wild_bootstrap",
             level: float = 0.05,
             mixing_weights: jnp.ndarray = None,
             return_p_val: bool = False
             ):
    psuedo_ys = generate_mixture_samples(key, credal_ys, mixing_weights)
    observed_mmd = compute_MMDsq(psuedo_ys, xs, kernel)

    # simulate the null
    if permutation_type == "permutation_test":
        joint_samples = jnp.concat([xs, psuedo_ys])
        permutations = vmap(lambda key: jax.random.permutation(key, joint_samples.shape[0]))(
            jax.random.split(key, num_permutations))
        simulated_mmds = vmap(lambda permutation: shuffle_cut_and_compute_mmd(joint_samples, permutation, kernel))(
            permutations)
        simulated_mmds = jnp.append(simulated_mmds, observed_mmd)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    elif permutation_type == "wild_bootstrap":
        h_gram = compute_h_gram(xs, psuedo_ys, kernel)
        simulated_mmds = compute_wild_bootstrap_samples(key, h_gram, num_permutations)
        critical_value = jnp.quantile(simulated_mmds, 1 - level)
    else:
        raise ValueError(f"Permutation type {permutation_type} not found.")

    if return_p_val:
        p_val = (simulated_mmds >= observed_mmd).mean()
        return p_val
    else:
        return (observed_mmd > critical_value).astype(jnp.int32)
