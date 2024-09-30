import pickle

import jax
import jax.numpy as jnp

from src.experiment_scripts.mnist.utils import build_credal_set_for_plausibility_test
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.plausibility.CredalPlausibilityTest import credal_plausibility_test
from src.testing_algorithms.plausibility.MMDQStarTest import mmdq_star_intersection_test
from src.testing_algorithms.plausibility.MMDQTest import mmdq_intersection_test
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == '__main__':
    load_data_path = "/home/$USER/projects/imprecise_testing/data/MNIST_features"
    master_key = jax.random.PRNGKey(0)
    num_samples = [50, 150, 250, 350, 500]
    num_samples = num_samples[::-1]
    powers = [0.5, 0.66, 0.75, 1.0]
    methods = ["MMDq", "MMDqstar"] + [f"CMMD({power})" for power in powers]
    nulls = [True, False]
    experiment_keys = jax.random.split(master_key, 500)

    configs = [(num_sample, method, null) for num_sample in num_samples for method in methods for null in nulls]
    save_path = ""

    for num_sample, method, null in configs:
        print(
            "Running experiment with",
            f"num_sample={num_sample}",
            f"method={method}",
            f"null={null}",
        )

        rejections, rejection = [], None

        for key in experiment_keys:
            if null == True:
                credal_x, credal_y = build_credal_set_for_plausibility_test(
                    load_data_path, [1, 3, 7], [1, 3, 9], num_sample)
            else:
                credal_x, credal_y = build_credal_set_for_plausibility_test(
                    load_data_path, [1, 8, 7], [2, 0, 9], num_sample)

            if method == "MMDq":
                rejection = mmdq_intersection_test(key, credal_x, credal_y, GaussianKernel(10.0))
            elif method == "MMDqstar":
                rejection = mmdq_star_intersection_test(key, credal_x, credal_y, GaussianKernel(10.0),
                                                        eps_power=-1 / 15)
            elif "CMMD" in method:
                for power in powers:
                    if method == f"CMMD({power})":
                        split_ratio = compute_root_split_ratio(num_sample, power)
                        rejection = credal_plausibility_test(key, credal_x, credal_y, GaussianKernel(10.0), split_ratio)
            if rejection is not None:
                rejections.append(rejection)

        rejection_rate = jnp.mean(jnp.array(rejections))
        print("Number of valid optimisation:", len(rejections))
        print("Rejection rate:", rejection_rate)
        config_name = f"num_sample={num_sample}_method={method}_null={null}.pkl"
        result = [
            method, num_sample, null, rejection_rate
        ]
        with open(save_path + config_name, 'wb') as f:
            pickle.dump(result, f)
