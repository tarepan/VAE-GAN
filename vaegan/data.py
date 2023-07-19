"""Data."""

import random

import torch
from torch import Tensor, tensor, stack      # pylint:disable=no-name-in-module
from torchvision import datasets, transforms # pyright:ignore[reportMissingTypeStubs]


def _load(indice: list[int], mnist: datasets.MNIST) -> list[tuple[Tensor, int]]:
    """Load MNIST items."""
    return list(map(lambda idx: mnist[idx], indice))


def _collate(items: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Collate MNIST items to a batch.
    
    Args:
        items :: (image::Tensor[1, X=28, Y=28], digit::int,)[]
    Returns:
        images :: (B, 1, X=28, Y=28) - Digit images
        digits :: (B,)               - Digit numbers
    """
    images =  stack(list(map(lambda image_digit: image_digit[0], items)))
    digits = tensor(list(map(lambda image_digit: image_digit[1], items)), dtype=torch.long)

    return images, digits


def _transfer(batch: tuple[Tensor, Tensor], device: str) -> tuple[Tensor, Tensor]:
    """Transfer a MNIST batch to the device."""
    return (batch[0].to(device), batch[1].to(device),)


class MNISTFixedOnMemory:
    """MNIST loader with on-memory fixed (non-shuffled) batch."""

    def __init__(self, batch_size: int, root: str = ".", download: bool = False, initial_shuffle: bool = False, device: str = "cpu"):
        """
        Args:
            batch_size
            root
            download
        """

        # Yield (image::Tensor[1, X=28, Y=28], digit::int,)
        mnist = datasets.MNIST(root, download=download, transform=transforms.ToTensor())

        # Prepare item indice
        n_item = len(mnist)
        item_indice = list(range(n_item))
        if initial_shuffle:
            random.shuffle(item_indice)

        # Batch-nize with drop last
        n_batch = n_item // batch_size
        item_indice = item_indice[ : n_batch * batch_size]
        item_indice_per_batch = [item_indice[idx_batch * batch_size : (idx_batch+1) * batch_size] for idx_batch in range(n_batch)]

        self._batches = list(map(lambda indice: _transfer(_collate(_load(indice, mnist)), device), item_indice_per_batch))

    def __iter__(self):
        return iter(self._batches)
