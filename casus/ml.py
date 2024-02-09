from jaxtyping import Array, ArrayLike
import equinox as eqx
from casus.encoders import RandomProjection, RFF
from casus import Fourier, MAP, HV
from casus.ops import rearrange

__all__ = ["Centroid"]


class Centroid(eqx.Module):
    """
    Trains a centroid classifier using the provided encoder.
    Learns prototypes for each class and for prediction, the closest
    prototype is selected.

    Args:
        num_classes: int
        encoder: RandomProjection | RFF

    """

    centroids: HV
    encoder: RandomProjection | RFF

    def __init__(self, num_classes: int, encoder: RandomProjection | RFF):
        super().__init__()

        self.encoder = encoder

        if isinstance(encoder, RandomProjection):
            _dim = encoder.projection.shape[0]
            self.centroids = MAP.empty((num_classes, _dim))
        elif isinstance(encoder, RFF):
            _dim = encoder.projection.shape[0]
            self.centroids = Fourier.empty((num_classes, _dim))
        else:
            raise ValueError("Unknown encoder type")

    def __call__(self, x: Array) -> ArrayLike:
        _x_hv = self.encoder(x)
        _preds = _x_hv.dota(self.centroids)  # type: ignore

        return _preds

    def train(self, x: Array, y: Array):
        _x_hv = self.encoder(x)
        _y_hv = self.centroids[y]

        _nc = self.centroids.shape[0]

        _updates = [_x_hv[y == c].set() for c in range(_nc)]
        _updates = rearrange(_updates, "nc 1 d -> nc d")
        new_centroids = self.centroids + _updates

        def where(m):
            return m.centroids

        new_model = eqx.tree_at(where, self, new_centroids)

        return new_model
