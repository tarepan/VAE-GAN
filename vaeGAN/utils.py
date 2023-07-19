"""Utilities."""

import torch
from torch import Tensor


def idx2onehot(indice: Tensor, n_class: int, idx2: None | Tensor, alpha: float):
    """Class index to one-hot vector.
    
    Args:
        indice :: (B,) - Class indice

        alpha          - Label breanding rate
    """

    indice = indice.cpu()

    # idx :: (B,) -> (B, 1)
    if indice.dim() == 1:
        indice = indice.unsqueeze(1)

    try:
        ans: list[Tensor] = []
        for idx_item in range(indice.cpu().data.numpy().size):
            # (Feat=n_class,)
            arr = torch.zeros(n_class)
            arr.scatter_(1, indice[idx_item], alpha)
            arr.scatter_(1, idx2, 1-alpha)
            ans.append(arr)
        # (B, Feat)
        onehot = torch.tensor(ans)
    except Exception:
        onehot = torch.zeros(indice.size(0), n_class)
        onehot.scatter_(1, indice, 1)

    return onehot.cuda()
