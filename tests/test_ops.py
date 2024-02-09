import pytest as pt
from casus import MAP, Fourier
from casus import rearrange, pack
import jax
import jax.numpy as jnp


@pt.mark.hv
def test_basic():
    a = MAP((10, 10))
    b = MAP((10, 10))

    c = a + b
    assert c.shape == (10, 10)
    assert isinstance(c, MAP)

    @jax.jit
    def f(x, y):
        return x + y

    _ = f(a, b)

    subb = a[0]
    assert subb.shape == (10,)
    assert isinstance(subb, MAP)

    subb = a[0:2]
    assert subb.shape == (2, 10)
    assert isinstance(subb, MAP)

    t = rearrange(a, "(n c) d -> c n d", c=2)
    assert t.shape == (2, 5, 10)

    _a = MAP((100, 200, 3))
    _b = MAP((100, 200))
    _c, ps = pack([_a, _b], "h w *")

    assert _c.shape == (100, 200, 4), f"{ps}"

    _a = MAP((10, 4096))
    _b = MAP((10, 4096))

    _c = _a.csim(_b)
    assert _c.shape == (10,)
    assert jnp.isclose(jnp.mean(_c), 0.5, atol=1e-1), f"{jnp.mean(_c)}"

    _c = _a.csima(_b)
    assert _c.shape == (10, 10)


@pt.mark.hv
def test_fourier():
    _a = Fourier((10, 4096))
    _b = Fourier((10, 4096))

    _c = _a.csim(_b)
    assert _c.shape == (10,)
    assert jnp.isclose(jnp.mean(_c), 0.0, atol=1e-1), f"{jnp.mean(_c)}"

    _c = _a.dota(_b)
    assert _c.shape == (10, 10)
