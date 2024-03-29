from .base import HV
import jax.numpy as jnp
from jax import core
from jaxtyping import Array, ArrayLike
from typing import Sequence
import equinox as eqx
import quax
from einops import einsum, reduce
import numpy as np
import jax

__all__ = ["Fourier"]


def _fourier_add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x + y


def _fourier_mul(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x * y


def _fourier_div(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x / y


class Fourier(HV):
    """
    Implements the Multiply-Add-Permute VSA type.
    Elements are real valued numbers and can optionally be quantized

    Args:
        shape: The shape of the Fourier
        data(optional): The data of the Fourier

    Either one of shape/data must be provided
    """

    def __init__(self, shape: tuple[int, ...] | None = None, **kwargs):
        if "array" in kwargs:
            _data = kwargs["array"]
        elif shape is None:
            raise ValueError("shape must be provided if data is not")
        else:
            _data = self._init(shape)

        super().__init__(_data)

    def _init(self, shape: tuple[int, ...]):
        vecs = np.random.normal(size=shape)

        return vecs

    @classmethod
    def empty(cls, shape) -> "Fourier":
        return cls(array=jnp.zeros(shape))

    @staticmethod
    def default(
        primitive: core.Primitive,
        values: Sequence[ArrayLike],
        params: dict,
    ):
        raw_values: list[ArrayLike] = []
        for value in values:
            if eqx.is_array_like(value):
                raw_values.append(value)
            elif isinstance(value, Fourier):
                raw_values.append(value.array)
            elif isinstance(value, int | float):
                raise ValueError(
                    "Operations between Fourier and scalar values are not supported."
                )
            else:
                assert False  # should never happen
        # print(raw_values, primitive, **params)
        out = primitive.bind(*raw_values, **params)
        if primitive.multiple_results:
            return [Fourier(x) for x in out]
        else:
            return Fourier(array=out)

    @jax.jit
    def __add__(self, other: "Fourier") -> "Fourier":
        return quax.quaxify(_fourier_add)(self, other)  # type: ignore

    @jax.jit
    def __sub__(self, other: "Fourier") -> "Fourier":
        return quax.quaxify(_fourier_add)(self, -other)  # type: ignore

    @jax.jit
    def __mul__(self, other: "Fourier") -> "Fourier":
        return quax.quaxify(_fourier_mul)(self, other)  # type: ignore

    @jax.jit
    def __truediv__(self, other: "Fourier") -> "Fourier":
        return quax.quaxify(_fourier_div)(self, other)  # type: ignore

    def __neg__(self) -> "Fourier":
        res = quax.quaxify(_fourier_mul)(self, -1)  # type: ignore
        return res

    def __invert__(self) -> "Fourier":
        res = 1 / self.array
        return Fourier(array=res)

    def __rshift__(self, shifts: int) -> "Fourier":
        return Fourier(array=jnp.roll(self.array, shifts, axis=-1))

    def __lshift__(self, shifts: int) -> "Fourier":
        return Fourier(array=jnp.roll(self.array, -shifts, axis=-1))

    def __getitem__(self, item: int | slice) -> "Fourier":
        return Fourier(array=self.array[item])

    @jax.jit
    def dot(self, other: "Fourier") -> Array:
        _dot = einsum(self.array, other.array, "i j,i j->i")

        return _dot

    @jax.jit
    def dota(self, other: "Fourier") -> Array:
        _dotm = einsum(self.array, other.array, "m d, n d->m n")

        return _dotm

    @jax.jit
    def csim(self, other: "Fourier") -> Array:
        _a = self.array / jnp.linalg.norm(self.array, axis=-1, keepdims=True)
        _b = other.array / jnp.linalg.norm(other.array, axis=-1, keepdims=True)
        csim = einsum(_a, _b, "i j,i j->i")

        return csim

    @jax.jit
    def csima(self, other: "Fourier") -> Array:
        _a = self.array / jnp.linalg.norm(self.array, axis=-1, keepdims=True)
        _b = other.array / jnp.linalg.norm(other.array, axis=-1, keepdims=True)
        csim = einsum(_a, _b, "m d,n d->m n")

        return csim

    @jax.jit
    def set(self) -> "Fourier":
        _res = reduce(self.array, "i j-> 1 j", "sum")
        return Fourier(array=_res)

    @jax.jit
    def mbind(self) -> "Fourier":
        _res = reduce(self.array, "i j-> 1 j", "prod")
        return Fourier(array=_res)
