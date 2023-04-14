import typing as tp

import numpy as np

import torch
import torchvision.transforms as T


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

TO_KEEP = [True] * 33 * 3
for i in range(len(TO_KEEP)):
    if i < 11 * 3 or i >= 25 * 3:
        TO_KEEP[i] = False
limit_shape = 115


class TrainTransforms:
    def __init__(self, to_keep: tp.Optional[tp.Sequence] = None, device: str = default_device) -> None:
        if to_keep is None:
            to_keep = TO_KEEP

        self.transforms = T.Compose([
            FilterIndex(to_keep=to_keep),
            NumpyToTensor(device=device),
            LimitShape(shape=limit_shape),
            NormalRandom(std=30.0),
            NormalizePoints(dim=1),
        ])

    def __call__(self, data: tp.Any) -> tp.Any:
        return self.transforms(data)


class TestTransforms:
    def __init__(self, to_keep: tp.Optional[tp.Sequence] = None, device: str = default_device) -> None:
        if to_keep is None:
            to_keep = TO_KEEP

        self.transforms = T.Compose([
            FilterIndex(to_keep=to_keep),
            NumpyToTensor(device=device),
            LimitShape(shape=limit_shape),
            NormalizePoints(dim=1),
        ])

    def __call__(self, data: tp.Any) -> tp.Any:
        return self.transforms(data)


class LabelsTransforms:
    def __init__(self, device: str = default_device) -> None:
        self.transforms = T.Compose([
            NumpyToLongTensor(device=device),
            LimitShape(shape=limit_shape),
        ])

    def __call__(self, data: tp.Any) -> tp.Any:
        return self.transforms(data)


class FilterIndex:
    def __init__(self, to_keep: tp.Sequence) -> None:
        self.to_keep = to_keep

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        return tensor[..., self.to_keep]


class NumpyToTensor:
    def __init__(self, device: str = default_device) -> None:
        self.device = device

    def __call__(self, tensor: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(tensor).float().to(self.device)


class NumpyToLongTensor:
    def __init__(self, device: str = default_device) -> None:
        self.device = device

    def __call__(self, tensor: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(tensor).long().to(self.device)


class LimitShape:
    def __init__(self, shape: int) -> None:
        self.shape = shape

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:self.shape]


class NormalizeOverDim:
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean(dim=self.dim).unsqueeze(self.dim)
        std = tensor.std(dim=self.dim).unsqueeze(self.dim)
        return (tensor - mean) / std


class NormalizePoints:
    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_points = tensor.reshape(tensor.shape[0], -1, 3)
        points_mean = tensor_points.mean(self.dim).unsqueeze(self.dim)
        points_std = tensor_points.std(self.dim).unsqueeze(self.dim)
        return ((tensor_points - points_mean) / points_std).reshape_as(tensor)


class UniformRandom:
    def __init__(self, bound: float) -> None:
        self.bound = bound

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 2 * self.bound * (torch.rand_like(tensor) - 0.5)


class NormalRandom:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.std * torch.randn_like(tensor)


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
