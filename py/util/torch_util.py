from typing import Dict, Tuple, Optional

import torch

Shape = Tuple[int, ...]


def apply_mask(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return tensor
    return tensor[mask]
