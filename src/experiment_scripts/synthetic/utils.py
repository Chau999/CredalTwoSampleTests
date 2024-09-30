import jax
from jax import numpy as jnp, Array
from jax.random import PRNGKey

from src.simulation.multivariate_gaussian import generate_corner_gaussian_samples
from src.simulation.multivariate_student import generate_corner_student_samples
from src.simulation.utils import generate_mixture_samples
from src.simulation.utils import initialise_moments_on_sphere
from src.testing_algorithms.plausibility.CredalDoubleDipPlausibilityTest import \
    two_staged_double_dip_split_inf_inf_mmd_test
from src.testing_algorithms.plausibility.CredalPlausibilityTest import credal_plausibility_test
from src.testing_algorithms.plausibility.MMDQStarTest import mmdq_star_intersection_test
from src.testing_algorithms.plausibility.MMDQTest import mmdq_intersection_test
from src.testing_algorithms.specification.CredalDoubleDipSpecificationTest import credal_double_dip_specification_test
from src.testing_algorithms.specification.CredalSpecificationTest import credal_specification_test
from src.testing_algorithms.specification.MMDQStarTest import mmdqstar_test
from src.testing_algorithms.specification.MMDQTest import mmdq_test
from src.testing_algorithms.specification.MMDTest import mmd_test


def generate_student_data_with_equal_credal_set(key: PRNGKey,
                                                num_extremes_x: int,
                                                num_extremes_y: int,
                                                num_samples: int,
                                                null: int | bool,
                                                df: int = 1,
                                                d: int = 2,
                                                ) -> tuple[Array, Array]:
    """Generate Student data for plausibility test."""
    key1, key2, key3 = jax.random.split(key, 3)
    means_xs, cov_xs = initialise_moments_on_sphere(num_extremes=num_extremes_x, radius=1.0, dim=d, key=PRNGKey(0))
    means_ys, cov_ys = initialise_moments_on_sphere(num_extremes=num_extremes_y, radius=1.0, dim=d, key=PRNGKey(0))
    if null:
        corner_samples_x = generate_corner_student_samples(key1, means_xs, num_samples, df=df)
        corner_samples_y = generate_corner_student_samples(key2, means_ys, num_samples, df=df)
    else:
        corner_samples_x = generate_corner_student_samples(key1, means_xs, num_samples, df=df)
        corner_samples_y = generate_corner_gaussian_samples(key2, means_ys, cov_ys, num_samples)

    return corner_samples_x, corner_samples_y


def generate_student_data_for_plausibility_test_with_overlapping_credal_sets(key: PRNGKey,
                                                                             num_extremes_x: int,
                                                                             num_extremes_y: int,
                                                                             num_samples: int,
                                                                             null: int | bool,
                                                                             df: int = 3,
                                                                             d: int = 2
                                                                             ) -> tuple[Array, Array]:
    """Generate Student data for plausibility test."""
    key1, key2, key3 = jax.random.split(key, 3)

    means_xs, cov_xs = initialise_moments_on_sphere(num_extremes=num_extremes_x, radius=1.0, dim=d, key=PRNGKey(0))
    means_ys, cov_ys = initialise_moments_on_sphere(num_extremes=num_extremes_y, radius=2.0, dim=d, key=PRNGKey(1))

    if null:
        corner_samples_x = generate_corner_student_samples(key1, means_xs, num_samples, df=df)
        means_ys = jnp.concat([means_ys[:-2, :], means_xs[-2:, :]], axis=0)
        corner_samples_y = generate_corner_student_samples(key2, means_ys, num_samples, df=df)
    else:
        corner_samples_x = generate_corner_student_samples(key1, means_xs, num_samples, df=df)
        corner_samples_y = generate_corner_gaussian_samples(key2, means_ys, cov_ys, num_samples)

    return corner_samples_x, corner_samples_y


def generate_student_data_for_inclusion_test(key: PRNGKey,
                                             num_extremes_outer: int,
                                             num_samples: int,
                                             mixing_weight_ls: list[jnp.ndarray],
                                             null: int | bool,
                                             df: int = 3,
                                             d: int = 2
                                             ) -> tuple[Array, Array]:
    """Generate Student data for set specification test."""
    key1, key2, key3 = jax.random.split(key, 3)
    means, cov = initialise_moments_on_sphere(num_extremes=num_extremes_outer, radius=2.0, dim=d)
    if null:
        samples_from_corners_temp = generate_corner_student_samples(key1, means, num_samples, df=df)
    else:
        samples_from_corners_temp = generate_corner_gaussian_samples(key1, means, cov, num_samples)

    samples_from_inner_corners = [
        generate_mixture_samples(key2, samples_from_corners_temp, mixing_weight)
        for mixing_weight in mixing_weight_ls
    ]

    samples_from_inner_corners = jnp.stack(samples_from_inner_corners, axis=2)
    samples_from_outer_corners = generate_corner_student_samples(key3, means, num_samples, df=df)

    return samples_from_inner_corners, samples_from_outer_corners


def generate_student_data_for_specification_test(key: PRNGKey,
                                                 num_extremes: int,
                                                 num_samples: int,
                                                 mixing_weights: jnp.ndarray,
                                                 null: int | bool,
                                                 df: int = 3,
                                                 d: int = 2
                                                 ) -> tuple[Array, Array]:
    """Generate Student data for pointwise specification test."""
    key1, key2, key3 = jax.random.split(key, 3)
    means, cov = initialise_moments_on_sphere(num_extremes=num_extremes, radius=1.0, dim=d)

    if null:
        samples_from_corners_temp = generate_corner_student_samples(key1, means, num_samples, df=df)
    else:
        samples_from_corners_temp = generate_corner_gaussian_samples(key1, means, cov, num_samples)

    samples = generate_mixture_samples(
        key=key2, credal_samples=samples_from_corners_temp, mixing_weights=mixing_weights)

    samples_from_corners = generate_corner_student_samples(key3, means, num_samples, df=df)

    return samples, samples_from_corners


def generate_student_data_for_specification_test_with_mixed_corners(key: PRNGKey,
                                                                    num_extremes: int,
                                                                    num_samples: int,
                                                                    mixing_weights: jnp.ndarray,
                                                                    null: int | bool,
                                                                    df: int = 3,
                                                                    d: int = 2
                                                                    ) -> tuple[Array, Array]:
    """Generate Student data for pointwise specification test."""
    """A few corners will be themselves a mixture of other corners"""
    key1, key2, key3, key4 = jax.random.split(key, 4)
    means, cov = initialise_moments_on_sphere(num_extremes=num_extremes - 1, radius=1.0, dim=d)

    if null:
        samples_from_corners_temp = generate_corner_student_samples(key1, means, num_samples, df=df)
        samples_from_corners_temp4 = generate_corner_student_samples(key4, means, num_samples, df=df)
        mixture_samples_for_corners = generate_mixture_samples(
            key=key1, credal_samples=samples_from_corners_temp4,
            mixing_weights=jnp.array([i / (num_extremes - 1) for i in range(num_extremes - 1)])).reshape(num_samples, d,
                                                                                                         1)
        samples_from_corners_temp = jnp.concatenate([samples_from_corners_temp, mixture_samples_for_corners], axis=2)
    else:
        samples_from_corners_temp = generate_corner_gaussian_samples(key1, means, cov, num_samples)
        samples_from_corners_temp4 = generate_corner_gaussian_samples(key4, means, cov, num_samples)
        mixture_samples_for_corners = generate_mixture_samples(
            key=key1, credal_samples=samples_from_corners_temp4,
            mixing_weights=jnp.array([i / (num_extremes - 1) for i in range(num_extremes - 1)])).reshape(num_samples, d,
                                                                                                         1)
        samples_from_corners_temp = jnp.concatenate([samples_from_corners_temp, mixture_samples_for_corners], axis=2)

    samples = generate_mixture_samples(
        key=key2, credal_samples=samples_from_corners_temp, mixing_weights=mixing_weights)

    samples_from_corners = generate_corner_student_samples(key3, means, num_samples, df=df)
    samples_from_corners4 = generate_corner_student_samples(key1, means, num_samples, df=df)
    mixture_samples_for_corners = generate_mixture_samples(
        key=key1, credal_samples=samples_from_corners4,
        mixing_weights=jnp.array([i / (num_extremes - 1) for i in range(num_extremes - 1)])).reshape(num_samples, d, 1)
    samples_from_corners = jnp.concatenate([samples_from_corners, mixture_samples_for_corners], axis=2)

    return samples, samples_from_corners


def get_testing_function_for_specification(testing_method: str, powers: list[float]):
    if testing_method == "MMDq":
        return mmdq_test
    elif testing_method == "MMDqstar":
        return lambda key, samples, samples_from_corners, kernel, level: mmdqstar_test(
            key, samples, samples_from_corners, kernel, eps_power=-1 / 15, level=level
        )
    elif testing_method in [f"2S-SplitMMD(n=m^{power})" for power in powers]:
        return credal_specification_test
    elif testing_method in [f"2S-DoubleDipMMD(n=m^{power})" for power in powers]:
        return credal_double_dip_specification_test
    elif testing_method == "MMD":
        return mmd_test
    else:
        return None


def get_testing_function_for_intersection(testing_method: str, powers: list[float]):
    if testing_method == "MMDq":
        return mmdq_intersection_test
    elif testing_method == "MMDqstar":
        return lambda key, samples_from_corners_1, samples_from_corners_2, kernel, level: mmdq_star_intersection_test(
            key, samples_from_corners_1, samples_from_corners_2, kernel, eps_power=-1 / 15, level=level
        )
    elif testing_method in [f"2S-SplitMMD(n=m^{power})" for power in powers]:
        return credal_plausibility_test
    elif testing_method in [f"2S-DoubleDipMMD(n=m^{power})" for power in powers]:
        return two_staged_double_dip_split_inf_inf_mmd_test
    else:
        return None
