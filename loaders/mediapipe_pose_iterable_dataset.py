import os
import typing as tp

import numpy as np
import torch

import loaders.utils as loaders_utils
from loaders import abstract_dataset


class MediapipePoseIterableDataset(abstract_dataset.AbstractIterableDataset):
    def __init__(
        self,
        samples: tp.List[str],
        label_map: tp.Dict[str, int],
        transforms: tp.Any = None,
        labels_transforms: tp.Any = None,
        batch_size: int = 1,
        base_fps: int = 30,
        target_fps: int = 30,
        with_rejection: bool = True,
    ) -> None:
        super().__init__(samples, label_map, transforms, labels_transforms, batch_size)

        self.base_fps = base_fps
        self.target_fps = target_fps
        self.with_rejection = with_rejection

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(sample).split('_')[1]

    def process_samples(
        self,
        file_path: str,
        batch_idx: int,
    ) -> tp.Generator[tp.Tuple[torch.Tensor, int], None, None]:
        label = self.label_map[self._extract_label(file_path)]

        trial_data = np.load(file_path, allow_pickle=True)
        points = trial_data[:, :-1]
        labels = (trial_data[:, -1] * label)

        for frame in loaders_utils.frequency_controller(self.target_fps / self.base_fps, len(trial_data)):
            input_tensor = self._transform_sample(points[frame])
            input_labels = self._transform_labels(labels[frame])
            yield input_tensor, input_labels
