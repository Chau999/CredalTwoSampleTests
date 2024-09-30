from functools import partial
from typing import Tuple, Any

import jax.numpy as jnp
from jax import Array, jit
from jaxopt import OSQP

from src.testing_algorithms.kernels import Kernel, gram_matrix
from src.testing_algorithms.mmd import compute_credal_embedding_matrix


@partial(jit, static_argnames=("kernel"))
def compute_infMMDsq_batch(credal_xs: Array, credal_ys: Array, kernel: Kernel) -> Tuple[
    Array, Array, Array]:
    """Compute the infMMDsq between credal samples and observed samples by casting the problem as a quadratic program

    :param credal_xs: credal sample of shape (num_sample, num_features, num_credal_samples)
    :param credal_ys: credal sample of shape (num_sample, num_features, num_credal_samples)
    :param kernel: kernel function
    :return: MMDsq value, estimated weights, error
    """

    K_xx_credal = compute_credal_embedding_matrix(credal_xs, credal_xs, kernel)
    K_yx_credal = compute_credal_embedding_matrix(credal_ys, credal_xs, kernel)
    K_yy_credal = compute_credal_embedding_matrix(credal_ys, credal_ys, kernel)
    Q = K_xx_credal + K_yy_credal - K_yx_credal - K_yx_credal.T
    c = jnp.zeros(Q.shape[0])
    A, b, G, h = set_up_quadratic_program_constraints(K_xx_credal.shape[0])

    qp = OSQP()
    sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h))
    param, error = sol.params.primal, sol.state.error
    MMDsq = jnp.dot(param, jnp.dot(Q, param))

    return MMDsq, param, error


@partial(jit, static_argnames=("kernel"))
def compute_infMMDsq(credal_xs: Array, ys: Array, kernel: Kernel) -> tuple[Array, Any]:
    """Compute the infMMDsq between credal samples and observed samples

    :param credal_xs: credal sample of shape (num_sample, num_features, num_credal_samples)
    :param ys: observed samples of shape (num_sample, num_features)
    :param kernel: kernel function
    :return: MMDsq value, estimated weights
    """
    K_xx_credal = compute_credal_embedding_matrix(credal_xs, credal_xs, kernel)
    batch_ys = jnp.stack([ys], axis=2)
    K_yx_credal = compute_credal_embedding_matrix(batch_ys, credal_xs, kernel)
    K_yy = gram_matrix(ys, ys, kernel)

    # set up the OSQP problem
    Q = 2.0 * K_xx_credal
    c = -2.0 * K_yx_credal[0]
    A, b, G, h = set_up_quadratic_program_constraints(K_xx_credal.shape[0])

    qp = OSQP()
    sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
    param = sol.primal
    MMDsq = jnp.dot(param, jnp.dot(Q, param)) / 2 + K_yy.mean() - jnp.dot(param, c)

    return MMDsq, param


@partial(jit, static_argnames=("kernel", "epoch"))
def compute_infinfMMDsq(credal_xs: Array, credal_ys: Array, kernel: Kernel, epoch: int = 10) -> tuple[
    list, list, Array]:
    """Compute the infinfMMDsq between credal sets using iterative minimisation

    :param credal_xs: credal sample of shape (num_sample, num_features, num_credal_samples)
    :param credal_ys: credal sample of shape (num_sample, num_features, num_credal_samples)
    :param kernel: kernel function
    :param epoch: number of iterations for the minimisation
    :return: estimated weights for x, estimated weights for y, objectives
    """

    def _compute_MMDsq(K_yy_credal, K_xx_credal, K_xy_credal, weight_x, weight_y):
        return jnp.dot(weight_y, jnp.dot(K_yy_credal, weight_y)) + jnp.dot(weight_x, jnp.dot(K_xx_credal,
                                                                                             weight_x)) - 2 * jnp.dot(
            weight_y, jnp.dot(K_xy_credal.T, weight_x))

    """Compute the inf-inf MMD between batches"""
    objectives, weight_ys, weight_xs = jnp.array([]), [], []
    K_xx_credal = compute_credal_embedding_matrix(credal_xs, credal_xs, kernel)
    K_xy_credal = compute_credal_embedding_matrix(credal_xs, credal_ys, kernel)
    K_yy_credal = compute_credal_embedding_matrix(credal_ys, credal_ys, kernel)

    initial_weights_y = jnp.array([i / K_yy_credal.shape[0] for i in range(K_yy_credal.shape[0])])
    weight_x = compute_quadratic_programming(2 * K_xx_credal, -2 * jnp.dot(K_xy_credal, initial_weights_y))
    MMDsq_0 = _compute_MMDsq(K_yy_credal, K_xx_credal, K_xy_credal, weight_x, initial_weights_y)
    objectives = jnp.append(objectives, MMDsq_0)
    weight_xs.append(weight_x)

    for rd in range(epoch):
        # Optimise over y
        weight_x = weight_xs[-1]
        weight_y = compute_quadratic_programming(2 * K_yy_credal, -2 * jnp.dot(K_xy_credal.T, weight_x))
        MMDsq = _compute_MMDsq(K_yy_credal, K_xx_credal, K_xy_credal, weight_x, weight_y)
        objectives = jnp.append(objectives, MMDsq)
        weight_ys.append(weight_y)

        # Optimise over x
        weight_y = weight_ys[-1]
        weight_x = compute_quadratic_programming(2 * K_xx_credal, -2 * jnp.dot(K_xy_credal, weight_y))
        MMDsq = _compute_MMDsq(K_yy_credal, K_xx_credal, K_xy_credal, weight_x, weight_y)
        objectives = jnp.append(objectives, MMDsq)
        weight_xs.append(weight_x)

    return weight_xs, weight_ys, objectives


def compute_quadratic_programming(Q: Array, c: Array):
    A, b, G, h = set_up_quadratic_program_constraints(Q.shape[0])
    qp = OSQP()
    sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
    return sol.primal


def set_up_quadratic_program_constraints(m: int) -> tuple[Array, Array, Array, Array]:
    A = jnp.array([[1.0] * m])
    b = jnp.array([1.0])
    G = -1.0 * jnp.eye(m)
    h = jnp.array([0.0] * m)

    return A, b, G, h
