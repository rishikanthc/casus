import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import ArrayLike, Array
from casus.hypervectors import Fourier, MAP
from einops import einsum

__all__ = ["RandomProjection", "RFF"]


class RandomProjection(eqx.Module):
    projection: Array = eqx.field(converter=jnp.asarray)

    def __init__(self, features: int, dimensions: int):
        super().__init__()
        self.projection = np.random.normal(size=(dimensions, features))  # type: ignore

    @jax.jit
    def _proj(self, x: ArrayLike) -> ArrayLike:
        _vec = einsum(x, self.projection, "b f,d f -> b d")

        return _vec

    def __call__(self, x: ArrayLike) -> MAP:
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)

        _res = self._proj(x)
        _quant = jnp.sign(_res)

        return MAP(array=_quant)


class RFF(eqx.Module):
    projection: Array = eqx.field(converter=jnp.asarray)
    bias: Array = eqx.field(converter=jnp.asarray)
    quantize: bool = eqx.field(static=True)

    def __init__(
        self,
        features: int,
        dimensions: int,
        bandwidth: ArrayLike,
        key: jnp.ndarray = jax.random.PRNGKey(0),
        quantize: bool = False,
    ):
        super().__init__()

        self.quantize = quantize

        if isinstance(bandwidth, float):
            bandwidth = jnp.eye(features) * bandwidth
        elif bandwidth.ndim == 1:  # type: ignore
            bandwidth = jnp.diag(bandwidth)

        _cov = features / bandwidth**2
        _inf_mask = jnp.isinf(_cov)
        _cov = jnp.where(_inf_mask, 0.0, _cov)
        _scale = jnp.sqrt(_cov)

        key, subkey = jax.random.split(key)
        self.projection = jax.random.normal(key, (dimensions, features)) @ _scale
        self.bias = jax.random.uniform(
            subkey, (dimensions,), minval=0, maxval=2 * jnp.pi
        )

    @jax.jit
    def _proj(self, x: Array) -> Array:
        _proj = einsum(x, self.projection, "b f, d f -> b d")
        _cos = jnp.cos(_proj + self.bias)
        _vec = _cos * jnp.sqrt(2 / self.projection.shape[0])

        return _vec

    def __call__(self, x: ArrayLike) -> Fourier:
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)

        _res = self._proj(x)

        if self.quantize:
            _res = jnp.sign(_res)

        return Fourier(array=_res)
