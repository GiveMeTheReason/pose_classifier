import os
import typing as tp

import numpy as np

import torch
from torch.utils.data import Dataset


class MediapipeIterDataset(Dataset):
    def __init__(
        self,
        samples: tp.List[str],
        label_map: tp.Dict[str, int],
        transforms: tp.Any = None,
        with_rejection: bool = True,
    ) -> None:
        self.file_paths = samples
        self.label_map = label_map
        self.transforms = transforms

        self.with_rejection = with_rejection

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        label = self.label_map[self._extract_label(file_path)]

        trial_data = np.load(file_path, allow_pickle=True)[:115]
        points = trial_data[:, :-1]
        labels = (trial_data[:, -1] * label).astype(np.int64)

        input_tensor = self._transform_sample(points)
        return input_tensor, labels

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(sample).split('_')[1]

    def _transform_sample(self, sample: np.ndarray) -> torch.Tensor:
        if self.transforms is not None:
            return self.transforms(sample)
        return torch.from_numpy(sample).float()
