import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.synthetic.utils import (generate_student_data_for_inclusion_test,
                                                    get_testing_function_for_specification)
from src.simulation.utils import runif_in_simplex
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == "__main__":
    LENGTHSCALE = 10
    LEVEL = 0.05

    num_samples = [50, 100, 200, 300, 400, 500, 750, 1000]
    num_outer_extremes = [3, 5, 10]
    num_inner_extremes = [3, 5, 10]
    powers = [0.66, 0.75, 1.0]
    methods = []
    methods += ["MMDqstar", "MMDq"]
    methods += [f"2S-SplitMMD(n=m^{power})" for power in powers]
    weight_configs = [0]
    nulls = [True, False]

    master_key = PRNGKey(0)
    configs = [(num_sample, num_outer_extreme, num_inner_extreme, weight_config, null, method)
               for num_sample in num_samples
               for num_outer_extreme in num_outer_extremes
               for num_inner_extreme in num_inner_extremes
               for weight_config in weight_configs
               for null in nulls
               for method in methods]
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
            runif_in_simplex(PRNGKey(weight_config), num_outer_extreme)
            for _ in range(num_inner_extreme)
        ]
        testing_function = get_testing_function_for_specification(method, powers)

        temp_result = []
        result = None

        for key in experiment_keys:
            samples_from_inner_corners, samples_from_outer_corners = generate_student_data_for_inclusion_test(
                key, num_outer_extreme, num_sample, mixing_weight_ls, null, df=3, d=10
            )
            penalty = []
            result_of_multiple_testing = 0
            for i in range(num_inner_extreme):
                samples = samples_from_inner_corners[:, :, i]
                if method == "MMD":
                    result = testing_function(
                        key, samples, samples_from_outer_corners, GaussianKernel(LENGTHSCALE),
                        level=LEVEL / num_inner_extreme, mixing_weights=mixing_weight_ls[i]
                    )
                elif method == "MMDq":
                    result = testing_function(
                        key, samples, samples_from_outer_corners, GaussianKernel(LENGTHSCALE),
                        level=LEVEL / num_inner_extreme
                    )
                elif method == "MMDqstar":
                    result = testing_function(
                        key, samples, samples_from_outer_corners, GaussianKernel(LENGTHSCALE),
                        level=LEVEL / num_inner_extreme
                    )
                elif "2S-SplitMMD" in method:
                    for power in powers:
                        if method == f"2S-SplitMMD(n=m^{power})":
                            split_ratio = compute_root_split_ratio(num_sample, power=power)
                            result = testing_function(
                                key, samples, samples_from_outer_corners, GaussianKernel(LENGTHSCALE),
                                split_ratio, num_permutations=500, level=LEVEL / num_inner_extreme)
                elif "2S-DoubleDipMMD" in method:
                    for power in powers:
                        if method == f"2S-DoubleDipMMD(n=m^{power})":
                            result = testing_function(
                                key, samples, samples_from_outer_corners, GaussianKernel(LENGTHSCALE), power,
                                num_permutations=500, level=LEVEL / num_inner_extreme)
                else:
                    raise ValueError(f"Method {method} not recognised.")

                if result:
                    result_of_multiple_testing = result
                    break
                if result is None:
                    penalty.append(1)

            if len(penalty) != 0:
                break

            if result_of_multiple_testing is not None:
                temp_result.append(result_of_multiple_testing)

        print(len(temp_result))
        rejection_rate = jnp.mean(jnp.array(temp_result))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"{method}_num_sample={num_sample}_num_outer_extreme={num_outer_extreme}_num_inner_extreme={num_inner_extreme}_weight_config={weight_config}_null={null}.pkl"
        result = [method, num_sample, num_outer_extreme, num_inner_extreme, weight_config, null, rejection_rate]
        with open(file_path_dir + config_name, "wb") as f:
            pickle.dump(result, f)
