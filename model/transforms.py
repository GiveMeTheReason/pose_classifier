import typing as tp

import mrob
import numpy as np

import torch
import torchvision.transforms as T


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainTransforms:
    def __init__(
        self,
        to_keep: tp.Sequence,
        shape_limit: int,
        device: str = default_device,
    ) -> None:
        self.transforms = T.Compose([
            SO3Random(),
            NumpyToTensor(device=device),
            LimitShape(shape_limit=shape_limit),
            NormalizePoints(dim=1),
            FilterIndex(to_keep=to_keep),
            NormalRandom(std=0.05),
        ])

    def __call__(self, data: tp.Any) -> tp.Any:
        return self.transforms(data)


class TestTransforms:
    def __init__(
        self,
        to_keep: tp.Sequence,
        shape_limit: int,
        device: str = default_device,
    ) -> None:
        self.transforms = T.Compose([
            SO3Random(),
            NumpyToTensor(device=device),
            LimitShape(shape_limit=shape_limit),
            NormalizePoints(dim=1),
            FilterIndex(to_keep=to_keep),
        ])

    def __call__(self, data: tp.Any) -> tp.Any:
        return self.transforms(data)


class LabelsTransforms:
    def __init__(
        self,
        shape_limit: int,
        device: str = default_device,
    ) -> None:
        self.transforms = T.Compose([
            NumpyToLongTensor(device=device),
            LimitShape(shape_limit=shape_limit),
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
    def __init__(self, shape_limit: int) -> None:
        self.shape_limit = shape_limit

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:self.shape_limit]


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
        normalized = (tensor_points - points_mean) / points_std
        return (normalized - normalized[:, -1:, :]).reshape_as(tensor)


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


class SO3Random:
    def __init__(
        self,
        std_x: float = 0,
        std_y: float = 0,
        std_z: float = 0,
    ) -> None:
        self.std_x = 0
        self.std_y = 0
        self.std_z = 0
        self.std_vector = (std_x, std_y, std_z)

    def _generate_matrix(self) -> np.ndarray:
        rand_gauss = np.random.rand(3) * self.std_vector
        return mrob.geometry.SO3(rand_gauss).R()

    def __call__(self, array: np.ndarray) -> np.ndarray:
        matrix_r = self._generate_matrix()
        rotated = array.reshape(array.shape[0], -1, 3) @ matrix_r.T
        return rotated.reshape(array.shape)


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
