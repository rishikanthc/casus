from .base import HV
import jax.numpy as jnp
from jax import core
from jaxtyping import Array, ArrayLike
from typing import Sequence
import equinox as eqx
import quax
from einops import einsum
import numpy as np


def _map_add(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x + y


def _map_mul(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x * y


def _map_div(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    return x / y


class MAP(HV):
    """
    Implements the Multiply-Add-Permute VSA type.
    Elements are real valued numbers and can optionally be quantized

    Args:
        shape: The shape of the MAP
        data(optional): The data of the MAP

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
            elif isinstance(value, MAP):
                raw_values.append(value.array)
            elif isinstance(value, int | float):
                raise ValueError(
                    "Operations between MAP and scalar values are not supported."
                )
            else:
                assert False  # should never happen
        # print(raw_values, primitive, **params)
        out = primitive.bind(*raw_values, **params)
        if primitive.multiple_results:
            return [MAP(x) for x in out]
        else:
            return MAP(array=out)

    def __add__(self, other: "MAP") -> "MAP":
        return quax.quaxify(_map_add)(self, other)  # type: ignore

    def __sub__(self, other: "MAP") -> "MAP":
        return quax.quaxify(_map_add)(self, -other)  # type: ignore

    def __mul__(self, other: "MAP") -> "MAP":
        return quax.quaxify(_map_mul)(self, other)  # type: ignore

    def __truediv__(self, other: "MAP") -> "MAP":
        return quax.quaxify(_map_div)(self, other)  # type: ignore

    def __neg__(self) -> "MAP":
        res = quax.quaxify(_map_mul)(self, -1)  # type: ignore
        return MAP(array=res)

    def __invert__(self) -> "MAP":
        return -self

    def __rshift__(self, shifts: int) -> "MAP":
        return MAP(array=jnp.roll(self.array, shifts, axis=-1))

    def __lshift__(self, shifts: int) -> "MAP":
        return MAP(array=jnp.roll(self.array, -shifts, axis=-1))

    def __getitem__(self, item: int | slice) -> "MAP":
        return MAP(array=self.array[item])

    def dot(self, other: "MAP") -> Array:
        _dot = einsum(self.array, other.array, "i j,i j->i")

        return _dot

    def dota(self, other: "MAP") -> Array:
        _dotm = einsum(self.array, other.array, "m d, n d->m n")

        return _dotm

    def csim(self, other: "MAP") -> Array:
        _a = self.array / jnp.linalg.norm(self.array, axis=-1, keepdims=True)
        _b = other.array / jnp.linalg.norm(other.array, axis=-1, keepdims=True)
        csim = einsum(_a, _b, "i j,i j->i")

        return csim

    def csima(self, other: "MAP") -> Array:
        _a = self.array / jnp.linalg.norm(self.array, axis=-1, keepdims=True)
        _b = other.array / jnp.linalg.norm(other.array, axis=-1, keepdims=True)
        csim = einsum(_a, _b, "m d,n d->m n")

        return csim
