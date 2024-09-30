import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.synthetic.utils import (get_testing_function_for_specification,
                                                    generate_student_data_with_equal_credal_set)
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == "__main__":
    LENGTHSCALE = 10.0
    LEVEL = 0.05

    num_samples = [50, 100, 200, 300, 400, 500, 750, 1000]
    num_outer_extremes = [3, 5, 10]
    num_inner_extremes = [3, 5, 10]
    powers = [0.66, 0.75, 1.0]
    methods = ["MMDq", "MMDqstar"]
    methods += [f"2S-SplitMMD(n=m^{power})" for power in powers]
    weight_configs = [0]
    nulls = [True, False]

    master_key = PRNGKey(1)
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

        testing_function = get_testing_function_for_specification(method, powers)

        temp_result = []
        result = None
        for key in experiment_keys:
            credal_set_x, credal_set_y = generate_student_data_with_equal_credal_set(
                key, num_outer_extreme, num_inner_extreme, num_samples=num_sample, null=null, df=3, d=10
            )
            num_multiple_testing = num_inner_extreme + num_outer_extreme
            result_of_multiple_testing = 0
            for i in range(num_multiple_testing):
                if i < num_inner_extreme:
                    samples = credal_set_x[:, :, i]
                    if method == "MMDq":
                        result = testing_function(
                            key, samples, credal_set_y, GaussianKernel(LENGTHSCALE),
                            level=LEVEL / num_multiple_testing
                        )
                    elif method == "MMDqstar":
                        result = testing_function(
                            key, samples, credal_set_y, GaussianKernel(LENGTHSCALE),
                            level=LEVEL / num_multiple_testing
                        )
                    elif "2S-SplitMMD" in method:
                        for power in powers:
                            if method == f"2S-SplitMMD(n=m^{power})":
                                split_ratio = compute_root_split_ratio(num_sample, power=power)
                                result = testing_function(
                                    key, samples, credal_set_y, GaussianKernel(LENGTHSCALE),
                                    split_ratio, num_permutations=500, level=LEVEL / num_multiple_testing)
                    elif "2S-DoubleDipMMD" in method:
                        for power in powers:
                            if method == f"2S-DoubleDipMMD(n=m^{power})":
                                result = testing_function(
                                    key, samples, credal_set_y, GaussianKernel(LENGTHSCALE), power,
                                    num_permutations=500, level=LEVEL / num_multiple_testing)
                    else:
                        raise ValueError(f"Method {method} not recognised.")

                else:
                    samples = credal_set_y[:, :, i - num_inner_extreme]
                    if method == "MMDq":
                        result = testing_function(
                            key, samples, credal_set_x, GaussianKernel(LENGTHSCALE),
                            level=LEVEL / num_multiple_testing
                        )
                    elif method == "MMDqstar":
                        result = testing_function(
                            key, samples, credal_set_x, GaussianKernel(LENGTHSCALE),
                            level=LEVEL / num_multiple_testing
                        )
                    elif "2S-SplitMMD" in method:
                        for power in powers:
                            if method == f"2S-SplitMMD(n=m^{power})":
                                split_ratio = compute_root_split_ratio(num_sample, power=power)
                                result = testing_function(
                                    key, samples, credal_set_x, GaussianKernel(LENGTHSCALE),
                                    split_ratio, num_permutations=500, level=LEVEL / num_multiple_testing)
                    elif "2S-DoubleDipMMD" in method:
                        for power in powers:
                            if method == f"2S-DoubleDipMMD(n=m^{power})":
                                result = testing_function(
                                    key, samples, credal_set_x, GaussianKernel(LENGTHSCALE), power,
                                    num_permutations=500, level=LEVEL / num_multiple_testing)
                    else:
                        raise ValueError(f"Method {method} not recognised.")

                if result:  # if reject once, then reject the whole test
                    result_of_multiple_testing = result
                    break

                if result is None:
                    break

            if result is not None:
                temp_result.append(result_of_multiple_testing)

        rejection_rate = jnp.mean(jnp.mean(jnp.array(temp_result)))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"{method}_num_sample={num_sample}_num_outer_extreme={num_outer_extreme}_num_inner_extreme={num_inner_extreme}_weight_config={weight_config}_null={null}.pkl"
        result = [method, num_sample, num_outer_extreme, num_inner_extreme, weight_config, null, rejection_rate]
        with open(file_path_dir + config_name, "wb") as f:
            pickle.dump(result, f)
