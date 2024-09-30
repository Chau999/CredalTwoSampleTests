from functools import partial

from jax import Array, jit, vmap

from src.testing_algorithms.kernels import Kernel, gram_matrix


@partial(jit, static_argnames=("kernel"))
def compute_h_gram(xs: Array, ys: Array, kernel: Kernel) -> Array:
    """Compute the h-gram matrix between two sets of vectors."""
    K_xx = gram_matrix(xs, xs, kernel)
    K_yy = gram_matrix(ys, ys, kernel)
    K_xy = gram_matrix(xs, ys, kernel)
    K_yx = gram_matrix(ys, xs, kernel)

    return K_xx + K_yy - K_xy - K_yx


@partial(jit, static_argnames=("kernel"))
def compute_MMDsq(xs: Array, ys: Array, kernel: Kernel) -> Array:
    """Compute the maximum mean discrepancy between two sets of vectors."""
    return compute_h_gram(xs, ys, kernel).mean()


@partial(jit, static_argnames=("kernel"))
def compute_inner_product_of_embedding(xs: Array, ys: Array, kernel: Kernel) -> Array:
    """Compute the inner product of the embeddings of two sets of vectors."""
    return gram_matrix(xs, ys, kernel).mean()


@partial(jit, static_argnames=("kernel"))
def compute_credal_embedding_matrix(batch_xs: Array, batch_ys: Array, kernel: Kernel) -> Array:
    """Compute the embedding matrix of a set of vectors."""
    vectorised_compute = vmap(vmap(compute_inner_product_of_embedding, in_axes=(None, 2, None)),
                              in_axes=(2, None, None))
    return vectorised_compute(batch_xs, batch_ys, kernel)
