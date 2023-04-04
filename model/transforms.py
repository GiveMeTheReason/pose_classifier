import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T


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


class NormalizeOverDim:
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, dim=self.dim)


class ExponentialSmoothing:
    def __init__(
        self,
        alpha: float,
        device: str = 'cpu' if not torch.cuda.is_available() else 'cuda:1',
    ) -> None:
        self.alpha = torch.tensor([alpha], device=device)
        self.prev_state = None

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.prev_state is None:
            self.prev_state = tensor.detach().clone().requires_grad_(False)
            return tensor
        tensor = self.alpha * tensor + (1 - self.alpha) * self.prev_state
        self.prev_state = tensor.detach().clone().requires_grad_(False)
        return tensor
