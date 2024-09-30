import pickle
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.random import PRNGKey

from src.simulation.utils import generate_mixture_samples


def load_digit(digit, load_data_path="../../../data/MNIST_features"):
    with open(load_data_path + f"/{digit}_features.pkl", "rb") as f:
        return jnp.array(pickle.load(f))


def build_credal_set_for_equality_test(load_data_path: str,
                                       digits_null: list[int],
                                       digits_alt: list[int],
                                       num_samples: int,
                                       null: bool):
    img_features = [load_digit(digit, load_data_path=load_data_path) for digit in range(10)]
    if null:
        credal_xs = []
        for digit in digits_null:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            credal_xs.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]
        credal_ys = []
        for digit in digits_null:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            credal_ys.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]

        return jnp.stack(credal_xs, axis=2), jnp.stack(credal_ys, axis=2)
    else:
        credal_xs = []
        for digit in digits_null:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            credal_xs.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]
        credal_ys = []
        for digit in digits_alt:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            credal_ys.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]

        return jnp.stack(credal_xs, axis=2), jnp.stack(credal_ys, axis=2)


def build_credal_set_for_inclusion_test(key: PRNGKey,
                                        load_data_path: str,
                                        digits_x: list[int],
                                        digits_y_for_alternative: Optional[list[int]],
                                        num_samples: int,
                                        mixing_weight_ls,
                                        null: bool):
    img_features = [load_digit(digit, load_data_path=load_data_path) for digit in range(10)]

    if null is True:
        inner_credal_sample, outer_credal_sample = [], []
        for mixing_weight in mixing_weight_ls:
            temp_credal_samples = []
            for digit in digits_x:
                temp = img_features[digit]
                indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
                temp = temp[indicies]
                temp_credal_samples.append(temp[:num_samples])
                img_features[digit] = temp[num_samples:]
            temp_credal_samples = jnp.stack(temp_credal_samples, axis=2)
            inner_credal_sample.append(generate_mixture_samples(key, temp_credal_samples, mixing_weight))
        inner_credal_sample = jnp.stack(inner_credal_sample, axis=2)
        for digit in digits_x:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            outer_credal_sample.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]
        outer_credal_sample = jnp.stack(outer_credal_sample, axis=2)

        return inner_credal_sample, outer_credal_sample

    else:
        inner_credal_sample, outer_credal_sample = [], []
        for mixing_weight in mixing_weight_ls:
            temp_credal_samples = []
            for digit in digits_y_for_alternative:
                temp = img_features[digit]
                indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
                temp = temp[indicies]
                temp_credal_samples.append(temp[:num_samples])
                img_features[digit] = temp[num_samples:]
            temp_credal_samples = jnp.stack(temp_credal_samples, axis=2)
            inner_credal_sample.append(generate_mixture_samples(key, temp_credal_samples, mixing_weight))
        inner_credal_sample = jnp.stack(inner_credal_sample, axis=2)
        for digit in digits_x:
            temp = img_features[digit]
            indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
            temp = temp[indicies]
            outer_credal_sample.append(temp[:num_samples])
            img_features[digit] = temp[num_samples:]
        outer_credal_sample = jnp.stack(outer_credal_sample, axis=2)

        return inner_credal_sample, outer_credal_sample


def build_credal_set_for_plausibility_test(load_data_path: str,
                                           digits_x: list[int],
                                           digits_y: list[int],
                                           num_samples: int,
                                           ):
    img_features = [load_digit(digit, load_data_path=load_data_path) for digit in range(10)]

    # create credal samples for x first
    credal_sample_x = []
    for digit in digits_x:
        temp = img_features[digit]
        indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
        temp = temp[indicies]
        credal_sample_x.append(temp[:num_samples])
        img_features[digit] = temp[num_samples:]

    credal_sample_y = []
    for digit in digits_y:
        temp = img_features[digit]
        indicies = np.random.choice(temp.shape[0], temp.shape[0], replace=False)
        temp = temp[indicies]
        credal_sample_y.append(temp[:num_samples])
        img_features[digit] = temp[num_samples:]

    credal_sample_x = jnp.stack(credal_sample_x, axis=2)
    credal_sample_y = jnp.stack(credal_sample_y, axis=2)

    return credal_sample_x, credal_sample_y


def build_credal_set_for_specification_test(key: PRNGKey,
                                            load_data_path: str,
                                            digits: list[int],
                                            replace_digit: Optional[int],
                                            num_samples: int,
                                            mixing_weights: Array,
                                            null: bool):
    img_features = [load_digit(digit, load_data_path=load_data_path) for digit in digits]

    # create credal samples first
    remaining_features = []
    credal_sample_temp = []
    for img_feature in img_features:
        indicies = np.random.choice(img_feature.shape[0], img_feature.shape[0], replace=False)
        img_feature = img_feature[indicies]  # shuffle
        credal_sample_temp.append(img_feature[:num_samples])
        remaining_features.append(img_feature[num_samples:])

    credal_sample_temp = jnp.stack(credal_sample_temp, axis=2)
    mixture_sample = generate_mixture_samples(key, credal_sample_temp, mixing_weights)

    if null:
        credal_sample = []
        for img_feature in remaining_features:
            credal_sample.append(img_feature[:num_samples])

        return mixture_sample, jnp.stack(credal_sample, axis=2)

    else:
        remaining_features[-1] = load_digit(replace_digit, load_data_path=load_data_path)
        credal_sample = []
        for img_feature in remaining_features:
            credal_sample.append(img_feature[:num_samples])

        return mixture_sample, jnp.stack(credal_sample, axis=2)
