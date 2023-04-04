import os
import typing as tp

import numpy as np
import torch

from loaders import abstract_dataset


class MediapipePoseDataset(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        samples: tp.List[str],
        label_map: tp.Dict[str, int],
        batch_size: int = 1,
        transforms: tp.Any = None,
        base_fps: int = 30,
        target_fps: int = 30,
        with_rejection: bool = True,
    ) -> None:
        super().__init__(samples, label_map, batch_size, transforms)

        self.base_fps = base_fps
        self.target_fps = target_fps

        self.with_rejection = with_rejection

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(sample).split('_')[1]

    def _transform_sample(self, sample: np.ndarray) -> torch.Tensor:
        sample = sample.reshape(-1, 3)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return torch.from_numpy(sample).float()

    def process_samples(
        self,
        path: str,
        batch_idx: int,
    ) -> tp.Generator[tp.Tuple[torch.Tensor, int], None, None]:
        label = self.label_map[self._extract_label(path)]
        # with open(os.path.join(path, 'label.txt')) as label_file:
        #     label_start, label_finish = map(int, label_file.readline().strip().split())
        label_start, label_finish = 45, 65

        trial_data = np.load(path, allow_pickle=True)

        current_frame = max(0, self.base_fps - self.target_fps)
        for frame, sample in enumerate(trial_data):
            current_frame += self.target_fps
            while current_frame >= self.base_fps:
                current_frame -= self.base_fps
                current_flg = (label_start <= frame <= label_finish)
                current_label = label * current_flg

                if not self.with_rejection and not current_flg:
                    continue
                elif not current_flg:
                    current_label = self.label_map['_rejection']

                sample = self._transform_sample(sample)
                yield sample, current_label
