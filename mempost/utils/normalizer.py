import torch
from typing import Optional


class Normalizer:
    """Normalize a tensor with training mean and standard deviation.

    Attributes:
        mean: Mean over the training dataset batch dimension.
        std: Standard deviation over the training dataset batch dimension.
        eps: Small float to avoid dividing by zero.
    """

    def __init__(self, dataset: torch.Tensor, eps: Optional[float] = 0.00001):
        super().__init__()
        self.mean = torch.mean(dataset, 0)
        self.std = torch.std(dataset, 0)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean
