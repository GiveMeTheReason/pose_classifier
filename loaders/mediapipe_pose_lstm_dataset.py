import os
import typing as tp

import numpy as np
import torch

from loaders import abstract_dataset


class MediapipePoseLSTMDataset(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        samples: tp.List[str],
        label_map: tp.Dict[str, int],
        batch_size: int = 1,
        transforms: tp.Any = None,
        with_rejection: bool = True,
    ) -> None:
        super().__init__(samples, label_map, batch_size, transforms)

        self.with_rejection = with_rejection

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(sample).split('_')[1]

    def _transform_sample(self, sample: np.ndarray) -> torch.Tensor:
        if self.transforms is not None:
            return self.transforms(sample)
        return torch.from_numpy(sample).float()

    def process_samples(
        self,
        path: str,
        batch_idx: int,
    ) -> tp.Generator[tp.Tuple[torch.Tensor, torch.Tensor], None, None]:
        label = self.label_map[self._extract_label(path)]

        trial_data = np.load(path, allow_pickle=True)[:115]
        points = trial_data[:, :-1]
        labels = (trial_data[:, -1] * label).astype(np.int64)

        input_tensor = self._transform_sample(points)
        yield input_tensor, labels
