import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


class ToDevice:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.device = device

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)


class NumpyToTensor:
    def __init__(self) -> None:
        pass

    def __call__(self, tensor: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(tensor).float()


class ReshapePoints:
    def __init__(self) -> None:
        pass

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, 3)


class RemoveMean:
    def __init__(self, dim: int = 0, with_zeros: bool = True) -> None:
        self.dim = dim
        self.with_zeros = with_zeros

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.with_zeros:
            return tensor - tensor.mean(dim=self.dim)
        return tensor - tensor.sum(dim=self.dim) / (tensor != 0).sum(dim=self.dim)


class NormalizeOverDim:
    def __init__(self, p: float = 2, dim: int = 0) -> None:
        self.p = p
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, p=self.p, dim=self.dim)


class ExponentialSmoothing:
    def __init__(
        self,
        alpha: float,
    ) -> None:
        self.alpha = alpha
        self.prev_state = torch.zeros(1)
        self.is_inited = False

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_inited:
            self.prev_state = tensor.detach().clone()
            self.is_inited = True
            return tensor
        tensor = self.alpha * tensor + (1 - self.alpha) * self.prev_state
        self.prev_state = tensor.detach().clone()
        return tensor
