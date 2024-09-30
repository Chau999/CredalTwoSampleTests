import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.mnist.utils import build_credal_set_for_specification_test
from src.simulation.utils import runif_in_simplex
from src.testing_algorithms.specification.CredalSpecificationTest import credal_specification_test
from src.testing_algorithms.specification.MMDQStarTest import mmdqstar_test
from src.testing_algorithms.specification.MMDQTest import mmdq_test
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == '__main__':

    load_data_path = "/home/$USER/projects/imprecise_testing/data/MNIST_features"
    master_key = jax.random.PRNGKey(0)
    num_samples = [50, 100, 200, 300, 400, 500]
    powers = [0.5, 0.66, 0.75, 1.0]
    methods = ["MMDq", "MMDqstar"] + [f"CMMD({power})" for power in powers]
    weight_configs = [0, 1, 2]
    nulls = [True, False]
    experiment_keys = jax.random.split(master_key, 500)

    configs = [(num_sample, method, weight_config, null) for num_sample in num_samples for method in methods for
               weight_config in weight_configs for null in nulls]
    save_path = ""

    for num_sample, method, weight_config, null in configs:
        print(
            "Running experiment with",
            f"num_sample={num_sample}",
            f"method={method}",
            f"weight_config={weight_config}",
            f"null={null}",
        )

        rejections, rejection = [], None
        mixing_weights = runif_in_simplex(PRNGKey(weight_config), num_extremes=3)
        for key in experiment_keys:
            mixture_sample, credal_sample = build_credal_set_for_specification_test(
                key, load_data_path, [1, 3, 7], 9, num_sample, mixing_weights, null)
            if method == "MMDq":
                rejection = mmdq_test(key, mixture_sample, credal_sample, GaussianKernel(10.0))
            elif method == "MMDqstar":
                rejection = mmdqstar_test(key, mixture_sample, credal_sample, GaussianKernel(10.0), eps_power=-1/15)
            elif "CMMD" in method:
                for power in powers:
                    if method == f"CMMD({power})":
                        split_ratio = compute_root_split_ratio(num_sample, power)
                        rejection = credal_specification_test(key, mixture_sample, credal_sample,
                                                              GaussianKernel(10.0), split_ratio)
            if rejection is not None:
                rejections.append(rejection)

        rejection_rate = jnp.mean(jnp.array(rejections))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"num_sample={num_sample}_method={method}_weight_config={weight_config}_null={null}.pkl"
        result = [
            method, num_sample, null, weight_config, rejection_rate
        ]

        with open(save_path + f"{config_name}", "wb") as f:
            pickle.dump(result, f)
