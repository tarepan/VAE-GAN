"""Test data modules."""

from torch import Tensor, tensor, equal # pylint:disable=no-name-in-module

from .data import _collate               # pyright:ignore[reportPrivateUsage]


def test_collate():
    """Test `_collate`."""

    # (Tensor[1,2,3], int) x5
    items: list[tuple[Tensor, int]] = [
        (tensor([[ 0,  1,  2,], [ 3,  4,  5]]), 0),
        (tensor([[ 0,  1,  2,], [ 3,  4,  5]]), 2),
        (tensor([[10, 11, 12,], [13, 14, 15]]), 3),
        (tensor([[ 0,  1,  2,], [ 3,  4,  5]]), 1),
        (tensor([[ 0,  1,  2,], [ 3,  4,  5]]), 5),
    ]
    images_gt = tensor([
        [[ 0,  1,  2,], [ 3,  4,  5]],
        [[ 0,  1,  2,], [ 3,  4,  5]],
        [[10, 11, 12,], [13, 14, 15]],
        [[ 0,  1,  2,], [ 3,  4,  5]],
        [[ 0,  1,  2,], [ 3,  4,  5]],
    ])
    digits_gt = tensor([0, 2, 3, 1, 5])

    batch = _collate(items)

    assert batch[0].size() == (5, 2, 3)
    assert equal(batch[0], images_gt)
    assert equal(batch[1], digits_gt)


# # passed @ 2023-07-20
# def test_onmemory_loader():
#     batch_size = 180
#     loader_panda = MNISTFixedOnMemory(batch_size, ".", download=True, initial_shuffle=False, device="cpu")

#     dataset_train = datasets.MNIST('.', download=True, transform=transforms.ToTensor())
#     train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True) # pyright:ignore[reportUnknownVariableType]

#     for b_panda, b_official in zip(loader_panda, train_loader):
#         assert equal(b_panda[0], b_official[0])
#         assert equal(b_panda[1], b_official[1])
