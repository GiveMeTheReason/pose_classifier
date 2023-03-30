import csv
import os
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
        super().__init__(samples, label_map, batch_size, transforms)

        self.base_fps = base_fps
        self.target_fps = target_fps

        self.with_rejection = with_rejection

    def _split_by_points(self, coords: tp.List[float]) -> tp.List[tp.List[float]]:
        step = 3
        return [coords[i:i+step] for i in range(0, len(coords), step)]

    def _extract_label(self, sample: str) -> str:
        return os.path.basename(os.path.dirname(os.path.dirname(sample)))

    def _transform_sample(self, sample: tp.List[str]) -> tp.List[tp.List[float]]:
        coords = [float(coord) for coord in sample[:99]]
        coords = self._split_by_points(coords)
        if self.transforms is not None:
            coords = self.transforms(coords)
        return coords

    def process_samples(
        self,
        path: str,
        batch_idx: int,
    ):
        # label = self.label_map[self._get_pose(path)]
        # with open(os.path.join(path, 'label.txt'), 'r') as label_file:
        #     label_start, label_finish = map(int, label_file.readline().strip().split())
        label = 1
        label_start, label_finish = 45, 90

        current_frame = max(0, self.base_fps - self.target_fps)

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', )
            headers = next(csv_reader, None)

            for frame, row in enumerate(csv_reader):
                current_frame += self.target_fps
                while current_frame >= self.base_fps:
                    current_frame -= self.base_fps
                    current_flg = (label_start <= frame <= label_finish)
                    current_label = label * current_flg

                    if not self.with_rejection and not current_flg:
                        continue
                    elif not current_flg:
                        current_label = self.label_map['no_gesture']

                    coords = self._transform_sample(row)
                    yield coords, current_label
