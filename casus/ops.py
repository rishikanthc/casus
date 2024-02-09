from jax.typing import ArrayLike
from casus import HV
import quax
from einops import (
    rearrange as einops_rearrange,
    reduce as einops_reduce,
    repeat as einops_repeat,
    pack as einops_pack,
    unpack as einops_unpack,
    einsum as einops_einsum,
)

__all__ = ["rearrange", "reduce", "repeat", "pack", "unpack", "einsum"]


def _rearrange(*args, **kwargs) -> ArrayLike:
    return einops_rearrange(*args, **kwargs)


def rearrange(*args, **kwargs) -> HV:
    return quax.quaxify(_rearrange)(*args, **kwargs)  # type: ignore


def _reduce(*args, **kwargs) -> ArrayLike:
    return einops_reduce(*args, **kwargs)


def reduce(*args, **kwargs) -> HV:
    return quax.quaxify(_reduce)(*args, **kwargs)  # type: ignore


def _repeat(*args, **kwargs) -> ArrayLike:
    return einops_repeat(*args, **kwargs)


def repeat(*args, **kwargs) -> HV:
    return quax.quaxify(_repeat)(*args, **kwargs)  # type: ignore


def _pack(*args, **kwargs):
    return einops_pack(*args, **kwargs)


def pack(*args, **kwargs):
    return quax.quaxify(_pack)(*args, **kwargs)  # type: ignore


def _unpack(*args, **kwargs):
    return einops_unpack(*args, **kwargs)


def unpack(*args, **kwargs):
    return quax.quaxify(_unpack)(*args, **kwargs)  # type: ignore


def _einsum(*args, **kwargs):
    return einops_einsum(*args, **kwargs)


def einsum(*args, **kwargs):
    return quax.quaxify(_einsum)(*args, **kwargs)  # type: ignore
