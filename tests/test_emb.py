import pytest as pt
from casus.encoders import RandomProjection, RFF
import numpy as np


@pt.mark.emb
def test_RandomProjection():
    encoder = RandomProjection(10, 2048)
    assert encoder.projection.shape == (10, 2048)


@pt.mark.emb
def test_RFF():
    encoder = RFF(10, 2048, 1.0)
    assert encoder.projection.shape == (2048, 10)
    assert encoder.bias.shape == (10,)

    encoder = RFF(1, 2048, 1.0)
    assert encoder.projection.shape == (2048, 1)

    data = np.random.normal(size=(2000, 1))
    assert data.shape == (2000, 1)

    _x = encoder(data)

    _eval = np.linspace(-5, 5, 1000)[:, None]
    _e = encoder(_eval)
