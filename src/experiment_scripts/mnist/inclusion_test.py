import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.mnist.utils import build_credal_set_for_inclusion_test
from src.simulation.utils import runif_in_simplex
from src.testing_algorithms.specification.CredalSpecificationTest import credal_specification_test
from src.testing_algorithms.specification.MMDQStarTest import mmdqstar_test
from src.testing_algorithms.specification.MMDQTest import mmdq_test
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == '__main__':
    load_data_path = "/home/$USER/projects/imprecise_testing/data/MNIST_features"
    num_samples = [50, 100, 200, 300, 400, 500]
    num_inner_extremes = [3]
    num_outer_extremes = [3]
    powers = [0.5, 0.66, 0.75, 1.0]
    methods = ["MMDq", "MMDqstar"] + [f"CMMD({power})" for power in powers]
    weight_configs = [0]
    nulls = [True, False]

    master_key = PRNGKey(0)
    configs = [
        (num_sample, num_outer_extreme, num_inner_extreme, weight_config, null, method)
        for num_sample in num_samples
        for num_outer_extreme in num_outer_extremes
        for num_inner_extreme in num_inner_extremes
        for weight_config in weight_configs
        for null in nulls
        for method in methods
    ]
    experiment_keys = jax.random.split(master_key, 500)
    file_path_dir = ""

    for num_sample, num_outer_extreme, num_inner_extreme, weight_config, null, method in configs:
        print(
            "Running experiment with",
            f"num_sample={num_sample}",
            f"num_outer_extreme={num_outer_extreme}",
            f"num_inner_extreme={num_inner_extreme}",
            f"weight_config={weight_config}",
            f"null={null}",
            f"method={method}",
        )

        weight_keys = jax.random.split(PRNGKey(weight_config), num_inner_extreme)
        mixing_weight_ls = [
            runif_in_simplex(weight_key, num_outer_extreme)
            for weight_key in weight_keys
        ]

        temp_result, rejection = [], None

        for key in experiment_keys:
            inner_credal_sample, outer_credal_sample = build_credal_set_for_inclusion_test(
                key, load_data_path, digits_x=[1, 3, 7], digits_y_for_alternative=[1, 3, 9],
                num_samples=num_sample, mixing_weight_ls=mixing_weight_ls, null=null)

            result_of_multiple_testing = 0
            penalty = []
            for i in range(num_inner_extreme):
                mixture_sample = inner_credal_sample[:, :, i]
                if method == "MMDq":
                    rejection = mmdq_test(key, mixture_sample, outer_credal_sample, GaussianKernel(10.0),
                                          level=0.05 / num_inner_extreme)
                elif method == "MMDqstar":
                    rejection = mmdqstar_test(
                        key, mixture_sample, outer_credal_sample, GaussianKernel(10.0), eps_power=-1 / 15,
                        level=0.05 / num_inner_extreme)
                elif "CMMD" in method:
                    for power in powers:
                        if method == f"CMMD({power})":
                            split_ratio = compute_root_split_ratio(num_sample, power)
                            rejection = credal_specification_test(key, mixture_sample, outer_credal_sample,
                                                                  GaussianKernel(10.0), split_ratio,
                                                                  level=0.05 / num_inner_extreme)
                if rejection:
                    result_of_multiple_testing = rejection
                    break
                if rejection is None:
                    penalty.append(1)

            if len(penalty) != 0:
                break

            if result_of_multiple_testing is not None:
                temp_result.append(result_of_multiple_testing)

        print(len(temp_result))
        rejection_rate = jnp.mean(jnp.array(temp_result))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"num_sample={num_sample}_num_outer_extreme={num_outer_extreme}_num_inner_extreme={num_inner_extreme}_weight_config={weight_config}_null={null}_method={method}.pkl"
        result = [
            method, num_sample, null, weight_config, rejection_rate
        ]
        with open(file_path_dir + config_name, "wb") as f:
            pickle.dump(result, f)
