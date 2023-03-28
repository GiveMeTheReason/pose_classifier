import typing as tp

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: tp.Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()

        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(output, labels)
