from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array, vmap

KernelLike = Callable[[Array, Array], Array]


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array) -> Array:
        pass


@dataclass(frozen=True, eq=True)
class GaussianKernel(Kernel):
    l: float

    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / self.l ** 2)


def gram_matrix(x1: Array, x2: Array, kernel: KernelLike) -> Array:
    """Compute the Gram matrix of two sets of vectors."""
    return vmap(lambda x1: vmap(lambda x2: kernel(x1, x2))(x2))(x1)


@jax.jit
def compute_median_heuristic(x1: Array, x2: Optional[Array] = None) -> Array:
    """Compute the median heuristic for the Gaussian kernel."""
    if x2 is None:
        xs = x1
    else:
        xs = jnp.concatenate([x1, x2], axis=0)

    distances = vmap(lambda xa: vmap(lambda xb: ((xa - xb) ** 2).sum() / 2)(xs))(xs)
    return jnp.sqrt(jnp.median(distances))
