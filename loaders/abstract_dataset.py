import abc
import itertools
import random
import typing as tp

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


TDataset = tp.TypeVar('TDataset', bound='AbstractDataset')
TIterableDataset = tp.TypeVar('TIterableDataset', bound='AbstractIterableDataset')


class AbstractDataset(abc.ABC, Dataset):

    def __init__(
        self,
        samples: tp.Sequence[tp.Any],
        label_map: tp.Dict[tp.Any, int],
        transforms: tp.Any = None,
        labels_transforms: tp.Any = None,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.transforms = transforms
        self.labels_transforms = labels_transforms

    def __len__(self) -> int:
        return len(self.samples)

    def _transform_sample(self, sample: tp.Any) -> tp.Any:
        if self.transforms is not None:
            return self.transforms(sample)
        return torch.from_numpy(sample).float()

    def _transform_labels(self, labels: tp.Any) -> tp.Any:
        if self.labels_transforms is not None:
            return self.labels_transforms(labels)
        return torch.from_numpy(labels).long()

    @abc.abstractmethod
    def _extract_label(self, sample: tp.Any) -> int:
        ...


class AbstractIterableDataset(AbstractDataset, IterableDataset):

    def __init__(
        self,
        samples: tp.Sequence[tp.Any],
        label_map: tp.Dict[tp.Any, int],
        transforms: tp.Any = None,
        labels_transforms: tp.Any = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(samples, label_map, transforms, labels_transforms)

        self.batch_size = batch_size

    @abc.abstractmethod
    def process_samples(
        self,
        sample: tp.Any,
        batch_idx: int,
    ) -> tp.Generator[tp.Tuple[tp.Any, int], None, None]:
        ...

    @classmethod
    def split_datasets(
        cls: tp.Type[TIterableDataset],
        batch_size: int = 1,
        max_workers: int = 1,
        **kwargs,
    ) -> tp.List[TIterableDataset]:
        num_workers: int = max(1, min(batch_size, max_workers))
        for n in range(max_workers, 1, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        batch_size = batch_size // num_workers

        return [cls(batch_size=batch_size, **kwargs) for _ in range(num_workers)]

    @property
    def shuffle_samples(self) -> tp.List[tp.Any]:
        return random.sample(self.samples, len(self.samples))

    def _get_stream(self, data_list: tp.List[str], batch_idx: int) -> tp.Iterable[tp.Any]:
        return itertools.chain.from_iterable(
            map(self.process_samples, data_list, itertools.repeat(batch_idx))
        )

    def get_streams(self) -> tp.Iterator[tp.Tuple[tp.Any]]:
        return zip(*[self._get_stream(
            self.shuffle_samples, batch_idx
        ) for batch_idx in range(self.batch_size)])

    def __iter__(self) -> tp.Iterator[tp.Tuple[tp.Any]]:
        return self.get_streams()
