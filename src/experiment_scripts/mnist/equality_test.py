import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.mnist.utils import build_credal_set_for_equality_test
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
    nulls = [True, False]

    master_key = PRNGKey(0)
    configs = [
        (num_sample, num_outer_extreme, num_inner_extreme, method, null)
        for num_sample in num_samples
        for num_outer_extreme in num_outer_extremes
        for num_inner_extreme in num_inner_extremes
        for method in methods
        for null in nulls
    ]
    experiment_keys = jax.random.split(master_key, 500)
    file_path_dir = ""

    for num_sample, num_outer_extreme, num_inner_extreme, method, null in configs:
        print(
            "Running experiment with",
            f"num_sample={num_sample}",
            f"num_outer_extreme={num_outer_extreme}",
            f"num_inner_extreme={num_inner_extreme}",
            f"method={method}",
            f"null={null}",
        )

        temp_result = []
        rejeciton = None
        for key in experiment_keys:
            credal_set_x, credal_set_y = build_credal_set_for_equality_test(
                load_data_path, [1, 3, 7], [1, 3, 9], num_sample, null=null
            )
            num_multiple_testing = num_inner_extreme + num_outer_extreme
            result_of_multiple_testing = 0
            for i in range(num_multiple_testing):
                if i < num_inner_extreme:
                    samples = credal_set_x[:, :, i]
                    if method == "MMDq":
                        rejection = mmdq_test(key, samples, credal_set_y, GaussianKernel(10.0),
                                              level=0.05 / num_multiple_testing)
                    elif method == "MMDqstar":
                        rejection = mmdqstar_test(key, samples, credal_set_y, GaussianKernel(10.0),
                                                  eps_power=-1 / 15, level=0.05 / num_multiple_testing)
                    elif "CMMD" in method:
                        for power in powers:
                            if f"CMMD({power})" == method:
                                split_ratio = compute_root_split_ratio(num_sample, power)
                                rejection = credal_specification_test(key, samples, credal_set_y,
                                                                      GaussianKernel(10.0), split_ratio,
                                                                      level=0.05 / num_multiple_testing)
                else:
                    samples = credal_set_y[:, :, i - num_inner_extreme]
                    if method == "MMDq":
                        rejection = mmdq_test(key, samples, credal_set_x, GaussianKernel(10.0),
                                              level=0.05 / num_multiple_testing)
                    elif method == "MMDqstar":
                        rejection = mmdqstar_test(key, samples, credal_set_x, GaussianKernel(10.0),
                                                  eps_power=-1 / 15, level=0.05 / num_multiple_testing)
                    elif "CMMD" in method:
                        for power in powers:
                            if f"CMMD({power})" == method:
                                split_ratio = compute_root_split_ratio(num_sample, power)
                                rejection = credal_specification_test(key, samples, credal_set_x,
                                                                      GaussianKernel(10.0), split_ratio,
                                                                      level=0.05 / num_multiple_testing)
                if rejection:
                    result_of_multiple_testing = rejection
                    break

                if rejection is None:
                    break

            if rejection is not None:
                temp_result.append(result_of_multiple_testing)

        rejection_rate = jnp.mean(jnp.array(temp_result))
        print(f"Rejection rate: {rejection_rate}")
        config_name = f"num_sample={num_sample}_num_outer_extreme={num_outer_extreme}_num_inner_extreme={num_inner_extreme}_method={method}_null={null}.pkl"
        result = [method, num_sample, null, rejection_rate]

        with open(file_path_dir + config_name, "wb") as f:
            pickle.dump(result, f)
