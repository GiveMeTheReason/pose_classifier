import glob
import itertools
import os
import random
import typing as tp

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
        super.__init__(samples, label_map, batch_size, transforms)

        self.base_fps = base_fps
        self.target_fps = target_fps

        self.with_rejection = with_rejection

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(path)))

    def _transform_data(self, sample):
        ...

    def process_data(
        self,
        path: str,
        batch_idx: int,
    ):
        # TODO: implement loader
        raise NotImplementedError
        label = self.label_map[self._get_pose(path)]
        with open(os.path.join(path, 'label.txt'), 'r') as label_file:
            label_start, label_finish = map(int, label_file.readline().strip().split())

        current_frame = max(0, self.base_fps - self.target_fps)

        paths = sorted(glob.glob(os.path.join(path, self.data_type.value)))

        for pc_path in paths:
            idx = int(os.path.splitext(os.path.basename(pc_path))[0])
            current_frame += self.target_fps
            while current_frame >= self.base_fps:
                current_frame -= self.base_fps
                current_flg = (label_start <= idx <= label_finish)
                current_label = label * current_flg

                if not self.with_rejection and not current_flg:
                    continue
                elif not current_flg:
                    current_label = self.label_map['no_gesture']

                pc = pc_path
                if self.transforms is not None:
                    pc = self.transforms(pc, batch_idx)
                yield pc, current_label
