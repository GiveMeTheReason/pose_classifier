import os
import typing as tp

import numpy as np

import torch

from loaders import abstract_dataset


class MediapipePoseMappedDataset(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        samples: tp.List[str],
        label_map: tp.Dict[str, int],
        transforms: tp.Any = None,
        labels_transforms: tp.Any = None,
        with_rejection: bool = True,
    ) -> None:
        super().__init__(samples, label_map, transforms, labels_transforms)

        self.with_rejection = with_rejection

    def __getitem__(self, idx: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.samples[idx]
        label = self.label_map[self._extract_label(file_path)]

        trial_data = np.load(file_path, allow_pickle=True)
        points = trial_data[:, :-1]
        labels = (trial_data[:, -1] * label)

        input_tensor = self._transform_sample(points)
        input_labels = self._transform_labels(labels)
        return input_tensor, input_labels

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(sample).split('_')[1]
