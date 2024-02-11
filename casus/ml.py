import jax.numpy as jnp
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

    def __call__(self, x: Array) -> tuple[ArrayLike, HV]:
        _x_hv = self.encoder(x)
        _preds = _x_hv.csima(self.centroids)  # type: ignore

        return _preds, _x_hv

    def train(self, x: Array, y: Array) -> "Centroid":
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

    def itrain(self, x: Array, y: Array) -> "Centroid":
        _nc = self.centroids.shape[0]

        _scores, _x_hv = self(x)
        _preds = jnp.argmax(_scores, axis=-1)

        updates = []
        for c in range(_nc):
            j = jnp.where((_preds != y) & (_preds == c))[0]
            i = jnp.where((_preds != y) & (y == c))[0]
            _update = _x_hv[i].set() - _x_hv[j].set()
            updates.append(_update)

        _updates = rearrange(updates, "nc 1 d -> nc d")
        new_centroids = self.centroids + _updates
        new_model = eqx.tree_at(lambda m: m.centroids, self, new_centroids)

        return new_model

    # def itrain(self, x: Array, y: Array) -> "Centroid":
    #     _nc = self.centroids.shape[0]
    #     _scores, _x_hv = self(x)
    #     _preds = jnp.argmax(_scores, axis=-1)

    #     # This is a placeholder for whatever _x_hv[i].set() - _x_hv[j].set() is supposed to do
    #     def update_rule(c, _x_hv, _preds, y):
    #         j = jnp.where((_preds != y) & (_preds == c), 1, 0)
    #         i = jnp.where((_preds != y) & (y == c), 1, 0)
    #         # Placeholder operation, replace with actual logic
    #         _update = _x_hv * i[:, None] - _x_hv * j[:, None]

    #         _update = jnp.sum(_x_hv * i[:, None], axis=0) - jnp.sum(
    #             _x_hv * j[:, None], axis=0
    #         )
    #         return _update

    #     # Vectorize the update rule over classes
    #     updates = jax.vmap(update_rule, in_axes=(0, None, None, None))(
    #         jnp.arange(_nc), _x_hv, _preds, y
    #     )

    #     new_centroids = self.centroids + updates
    #     new_model = eqx.tree_at(lambda m: m.centroids, self, new_centroids)

    #     return new_model
