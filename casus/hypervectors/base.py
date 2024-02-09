import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp
import quax
from jax import core
from abc import ABC, abstractmethod, abstractclassmethod


class HV(quax.ArrayValue, ABC):
    array: jnp.ndarray = eqx.field(converter=jnp.asarray)

    def aval(self):
        shape = jnp.shape(self.array)
        dtype = jnp.result_type(self.array)
        return core.ShapedArray(shape, dtype)

    def materialise(self):
        return self.array

    @abstractclassmethod
    def empty(cls, shape) -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: "HV") -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: "HV") -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other: "HV") -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __truediv__(self, other: "HV") -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __invert__(self) -> "HV":
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: int | slice) -> "HV":
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.array)

    @abstractmethod
    def set(self) -> "HV":
        raise NotImplementedError

    @abstractmethod
    def mbind(self) -> "HV":
        raise NotImplementedError
