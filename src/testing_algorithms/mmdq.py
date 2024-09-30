"""Implementation of the MMDq estimators(https://arxiv.org/abs/2305.07549) in JAX."""

from functools import partial
from typing import List

import jax.numpy as jnp
from jax import jit

from src.testing_algorithms.kernels import Kernel, gram_matrix
from src.testing_algorithms.mmd import compute_MMDsq


def trim_input(xs: jnp.ndarray, ys: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # this is to ensure we have a multiple of 24 samples for the MMDq algorithms
    n = xs.shape[0]
    n_trim = (n // 24) * 24
    return xs[:n_trim], ys[:n_trim]


@partial(jit, static_argnames=("kernel"))
def compute_core(x1234: List[jnp.ndarray], y1234: List[jnp.ndarray], kernel: Kernel):
    x1, x2, x3, x4 = x1234
    y1, y2, y3, y4 = y1234

    Kxx13 = gram_matrix(x1, x3, kernel)
    Kxy42 = gram_matrix(x4, y2, kernel)
    Kxy24 = gram_matrix(x2, y4, kernel)
    Kyy13 = gram_matrix(y1, y3, kernel)

    return Kxx13 - Kxy42 - Kxy24 + Kyy13


@partial(jit, static_argnames=("kernel"))
def compute_h_core(x12: List[jnp.ndarray], y12: List[jnp.ndarray], kernel: Kernel):
    x1, x2 = x12
    y1, y2 = y12

    Kxx12 = gram_matrix(x1, x2, kernel)
    Kxy12 = gram_matrix(x1, y2, kernel)
    Kxy21 = gram_matrix(x2, y1, kernel)
    Kyy12 = gram_matrix(y1, y2, kernel)

    return Kxx12 - Kxy12 - Kxy21 + Kyy12


@partial(jit, static_argnames=("kernel"))
def compute_mmdq(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel) -> jnp.ndarray:
    """Compute the MMDq statistic between two sets of vectors."""
    xs, ys = trim_input(xs, ys)

    # compute standard mmd
    mmd = compute_MMDsq(xs, ys, kernel)

    # compute the numerator
    n, d = xs.shape
    space = n // 4

    x1, x2, x3, x4 = xs[:space], xs[space:2 * space], xs[2 * space:3 * space], xs[3 * space:]
    y1, y2, y3, y4 = ys[:space], ys[space:2 * space], ys[2 * space:3 * space], ys[3 * space:]

    numerator = compute_core([x1, x2, x3, x4], [y1, y2, y3, y4], kernel).mean()

    # compute the denominator
    space = n // 6
    x1, x2, x3, x4, x5, x6 = (xs[:space], xs[space:2 * space], xs[2 * space:3 * space], xs[3 * space:4 * space],
                              xs[4 * space:5 * space], xs[5 * space:])
    y1, y2, y3, y4, y5, y6 = (ys[:space], ys[space:2 * space], ys[2 * space:3 * space], ys[3 * space:4 * space],
                              ys[4 * space:5 * space], ys[5 * space:])

    term1 = compute_core([x1, x2, x3, x4], [y1, y2, y3, y4], kernel) * compute_core(
        [x1, x2, x5, x6], [y1, y2, y5, y6], kernel)
    term2 = compute_core([x3, x4, x1, x2], [y3, y4, y1, y2], kernel) * compute_core(
        [x3, x4, x5, x6], [y3, y4, y5, y6], kernel)
    term3 = compute_core([x5, x6, x3, x4], [y5, y6, y3, y4], kernel) * compute_core(
        [x5, x6, x1, x2], [y5, y6, y1, y2], kernel)

    denominator = (term1 + term2 + term3).mean() * (8 / 3) - 8 * (mmd ** 2)

    return jnp.sqrt(n) * numerator / jnp.sqrt(denominator)


def compute_mmdq_numerator(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel) -> jnp.ndarray:
    """Compute the numerator of the MMDq statistic."""

    # compute the numerator
    n, d = xs.shape
    space = n // 4

    x1, x2, x3, x4 = xs[:space], xs[space:2 * space], xs[2 * space:3 * space], xs[3 * space:]
    y1, y2, y3, y4 = ys[:space], ys[space:2 * space], ys[2 * space:3 * space], ys[3 * space:]

    return compute_core([x1, x2, x3, x4], [y1, y2, y3, y4], kernel).mean()


def compute_mmdq_denominator(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel) -> jnp.ndarray:
    """Compute the denominator of the MMDq statistic."""

    # compute standard mmd
    mmd = compute_MMDsq(xs, ys, kernel)

    # compute the denominator
    n, d = xs.shape
    space = n // 6
    x1, x2, x3, x4, x5, x6 = (xs[:space], xs[space:2 * space], xs[2 * space:3 * space], xs[3 * space:4 * space],
                              xs[4 * space:5 * space], xs[5 * space:])
    y1, y2, y3, y4, y5, y6 = (ys[:space], ys[space:2 * space], ys[2 * space:3 * space], ys[3 * space:4 * space],
                              ys[4 * space:5 * space], ys[5 * space:])

    term1 = compute_core([x1, x2, x3, x4], [y1, y2, y3, y4], kernel) * compute_core(
        [x1, x2, x5, x6], [y1, y2, y5, y6], kernel)
    term2 = compute_core([x3, x4, x1, x2], [y3, y4, y1, y2], kernel) * compute_core(
        [x3, x4, x5, x6], [y3, y4, y5, y6], kernel)
    term3 = compute_core([x5, x6, x3, x4], [y5, y6, y3, y4], kernel) * compute_core(
        [x5, x6, x1, x2], [y5, y6, y1, y2], kernel)

    return (term1 + term2 + term3).mean() * (8 / 3) - 8 * (mmd ** 2)


def compute_mmd_denominator(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel) -> jnp.ndarray:
    """Compute the denominator of the MMDq statistic."""

    n = xs.shape[0]

    # compute standard mmd
    mmd = compute_MMDsq(xs, ys, kernel)

    space = n // 3
    x1, x2, x3 = xs[:space], xs[space:2 * space], xs[2 * space:]
    y1, y2, y3 = ys[:space], ys[space:2 * space], ys[2 * space:]
    term3 = compute_h_core([x1, x2], [y1, y2], kernel) * compute_h_core([x1, x3], [y1, y3], kernel)
    term4 = compute_h_core([x2, x1], [y2, y1], kernel) * compute_h_core([x2, x3], [y2, y3], kernel)
    term5 = compute_h_core([x3, x1], [y3, y1], kernel) * compute_h_core([x3, x2], [y3, y2], kernel)

    return jnp.sqrt((term3 + term4 + term5).mean() * (4 / 3) - 4 * (mmd ** 2))


@partial(jit, static_argnames=("kernel"))
def compute_mmdq_star(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel, eps_power: float) -> jnp.ndarray:
    xs, ys = trim_input(xs, ys)
    n = xs.shape[0]
    # compute standard mmd
    mmd = compute_MMDsq(xs, ys, kernel)
    eps_n = n ** (eps_power)

    # compute the numerator
    mmdq_numerator = compute_mmdq_numerator(xs, ys, kernel)
    mmdq_star_numerator = mmdq_numerator * eps_n + mmd

    # compute the denominator
    mmdq_denominator = compute_mmdq_denominator(xs, ys, kernel)
    mmdq_star_denominator = mmdq_denominator * eps_n + compute_mmd_denominator(xs, ys, kernel)

    # return jnp.sqrt(n) * numerator / denominator
    return jnp.sqrt(n) * mmdq_star_numerator / mmdq_star_denominator


@partial(jit, static_argnames=("kernel"))
def compute_mmdq_split_star(xs: jnp.ndarray, ys: jnp.ndarray, kernel: Kernel, eps_power: float) -> jnp.ndarray:
    xs_train, xs_test = xs[:xs.shape[0] // 2], xs[xs.shape[0] // 2:]
    ys_train, ys_test = ys[:ys.shape[0] // 2], ys[ys.shape[0] // 2:]

    xs_train, ys_train = trim_input(xs_train, ys_train)
    xs_test, ys_test = trim_input(xs_test, ys_test)

    n_train = xs_train.shape[0]
    # compute standard mmd
    mmd = compute_MMDsq(xs_test, ys_test, kernel)
    eps_n = n_train ** (eps_power)

    # compute the numerator
    mmdq_numerator = compute_mmdq_numerator(xs_test, ys_test, kernel)
    mmdq_star_numerator = mmdq_numerator * eps_n + mmd

    # compute the denominator
    mmdq_denominator = compute_mmdq_denominator(xs_test, ys_test, kernel)
    mmdq_star_denominator = mmdq_denominator * eps_n + compute_mmd_denominator(xs_train, ys_train, kernel)

    # return jnp.sqrt(n) * numerator / denominator
    return jnp.sqrt(n_train) * mmdq_star_numerator / mmdq_star_denominator
