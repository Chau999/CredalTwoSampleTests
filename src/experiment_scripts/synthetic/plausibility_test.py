import pickle

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from src.experiment_scripts.synthetic.utils import (
    generate_student_data_for_plausibility_test_with_overlapping_credal_sets,
    get_testing_function_for_intersection
)
from src.testing_algorithms.kernels import GaussianKernel
from src.testing_algorithms.utils import compute_root_split_ratio

if __name__ == '__main__':
    FILE_PATH_DIR = ""
    LENGTHSCALE = 10.0
    LEVEL = 0.05

    num_samples = [50, 100, 200, 300, 400, 500, 750, 1000]
    num_extremes_x = [5, 10]
    num_extremes_y = [5, 10]
    powers = [0.5, 0.66, 0.75, 1.0]
    methods = ["MMDqstar"]
    methods += ["MMDq"]
    methods += [f"2S-SplitMMD(n=m^{power})" for power in powers]
    nulls = [True, False]

    master_key = PRNGKey(3)
    configs = [(num_sample, num_extreme_x, num_extreme_y, null, method)
               for num_sample in num_samples
               for num_extreme_x in num_extremes_x
               for num_extreme_y in num_extremes_y
               for null in nulls
               for method in methods]
    experiment_keys = jax.random.split(master_key, 500)

    for num_sample, num_extreme_x, num_extreme_y, null, method in configs:
        if num_extreme_x == num_extreme_y:
            print(
                "Running experiment with",
                f"num_sample={num_sample}",
                f"num_extreme_x={num_extreme_x}",
                f"num_extreme_y={num_extreme_y}",
                f"null={null}",
                f"method={method}",
            )

            testing_function = get_testing_function_for_intersection(method, powers=powers)
            temp_result = []
            rejection = 0
            for key in experiment_keys:
                credal_xs, credal_ys = generate_student_data_for_plausibility_test_with_overlapping_credal_sets(
                    key, num_extremes_x=num_extreme_x, num_extremes_y=num_extreme_y, num_samples=num_sample, null=null, df=3,
                    d=10
                )

                if method == "MMDq":
                    rejection = testing_function(
                        key, credal_xs, credal_ys, GaussianKernel(LENGTHSCALE)
                    )
                elif "2S-SplitMMD" in method:
                    for power in powers:
                        if method == f"2S-SplitMMD(n=m^{power})":
                            split_ratio = compute_root_split_ratio(num_sample, power=power)
                            rejection = testing_function(
                                key, credal_xs, credal_ys, GaussianKernel(LENGTHSCALE), split_ratio,
                                num_permutations=500
                            )
                elif "2S-DoubleDipMMD" in method:
                    for power in powers:
                        if method == f"2S-DoubleDipMMD(n=m^{power})":
                            rejection = testing_function(
                                key, credal_xs, credal_ys, GaussianKernel(LENGTHSCALE), power,
                                num_permutations=500
                            )
                elif method == "MMDqstar":
                    rejection = testing_function(
                        key, credal_xs, credal_ys, GaussianKernel(LENGTHSCALE), level=LEVEL
                    )
                else:
                    raise ValueError(f"Method {method} not found.")

                if rejection is not None:
                    temp_result.append(rejection)

            rejection_rate = jnp.mean(jnp.array(temp_result))
            print(len(temp_result))
            print(f"Rejection rate: {rejection_rate}")
            config_name = f"{method}_num_sample={num_sample}_num_extreme_x={num_extreme_x}_num_extreme_y={num_extreme_y}_null={null}.pkl"
            result = [method, num_sample, num_extreme_x, num_extreme_y, null, rejection_rate]

            with open(FILE_PATH_DIR + config_name, "wb") as f:
                pickle.dump(result, f)
