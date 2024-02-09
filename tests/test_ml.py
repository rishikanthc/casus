import jax.numpy as jnp
import fire
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from casus.encoders import RFF
from casus.ml import Centroid
import pytest as pt
from alive_progress import alive_bar


def scotts_rule(n: int, d: int):
    return jnp.power(n, -1 / (d + 4))


@pt.mark.ml
def test_Centroid(bw=310):
    """
    Test the centroid model on the MNIST dataset.

    Args:
        bw (int): The bandwidth of the RFF encoder.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = MNIST(root="data", train=True, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

    _bw = bw * jnp.eye(784)
    mnist_encoder = RFF(784, 2048, _bw)
    model = Centroid(10, mnist_encoder)
    tr_len = len(train_dl)

    with alive_bar(tr_len) as bar:
        for batch in train_dl:
            x, y = batch
            x = x.numpy().reshape(-1, 784)
            y = y.numpy()

            model = model.train(x, y)
            bar()

    with alive_bar(tr_len) as bar:
        avg_acc = 0.0
        for batch in train_dl:
            x, y = batch
            x = x.numpy().reshape(-1, 784)
            y = y.numpy()

            scores = model(x)
            preds = jnp.argmax(scores, axis=-1)
            acc = jnp.mean(preds == y)
            avg_acc += acc
            bar()

    avg_acc /= tr_len
    print(f"Average accuracy: {avg_acc}")

    assert avg_acc > 0.6, f"Accuracy too low: {avg_acc}"


def main():
    fire.Fire(test_Centroid)
