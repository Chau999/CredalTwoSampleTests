import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.synthetic.utils import generate_student_data_for_specification_test, \
    get_testing_function_for_specification
from src.simulation.utils import runif_in_simplex
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == '__main__':
    lengthscale = 10.0
    num_samples = [50, 100, 200, 300, 400, 500, 750, 1000]
    num_samples = num_samples[::-1]
    num_extremes = [5]
    powers = [0.66, 0.75, 1.0]  # the split for training and testing_algorithms data
    methods = (["MMDq", "MMDqstar"] +
               [f"2S-SplitMMD(n=m^{power})" for power in powers])
    weight_configs = [1]
    nulls = [True, False]

    configs = [(num_sample, num_extreme, weight_config, null, method)
               for num_sample in num_samples
               for num_extreme in num_extremes
               for weight_config in weight_configs
               for null in nulls
               for method in methods]

    master_key = PRNGKey(10)
    experiment_keys = jax.random.split(master_key, 500)
    FILE_PATH = ""

    for num_sample, num_extreme, weight_config, null, method in configs:
        print(
            "Running experiment with",
            f"num_sample={num_sample}",
            f"num_extreme={num_extreme}",
            f"weight_config={weight_config}",
            f"null={null}",
            f"method={method}",
        )

        mixing_weights = runif_in_simplex(PRNGKey(weight_config), num_extreme)
        print("Mixing weights: ", mixing_weights)
        testing_function = get_testing_function_for_specification(method, powers)

        temp_result = []
        result = None
        for i, key in enumerate(experiment_keys):
            samples, samples_from_corners = generate_student_data_for_specification_test(
                key, num_extreme, num_sample, mixing_weights, null, d=10, df=3
            )

            if "2S-SplitMMD" in method:
                for power in powers:
                    if method == f"2S-SplitMMD(n=m^{power})":
                        split_ratio = compute_root_split_ratio(num_sample, power=power)
                        result = testing_function(
                            key, samples, samples_from_corners, GaussianKernel(lengthscale), split_ratio,
                            num_permutations=500)
            elif "2S-DoubleDipMMD" in method:
                for power in powers:
                    if method == f"2S-DoubleDipMMD(n=m^{power})":
                        result = testing_function(
                            key, samples, samples_from_corners, GaussianKernel(lengthscale), power,
                            num_permutations=500)
            elif method == "MMDqstar":
                result = testing_function(
                    key, samples, samples_from_corners, GaussianKernel(lengthscale),
                    level=0.05
                )
            else:
                result = testing_function(key, samples, samples_from_corners, GaussianKernel(lengthscale))

            if result is not None:
                temp_result.append(result)

        rejection_rate = jnp.mean(jnp.array(temp_result))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"{method}_num_sample={num_sample}_num_extreme={num_extreme}_weight_config={weight_config}_null={null}.pkl"
        result = [
            method,
            num_sample,
            num_extreme,
            null,
            weight_config,
            rejection_rate
        ]

        with open(FILE_PATH + config_name, "wb") as f:
            pickle.dump(result, f)
