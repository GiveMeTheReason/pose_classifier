import itertools
import typing as tp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def split_datasets(
    cls: Dataset,
    batch_size: int = 1,
    max_workers: int = 1,
    **kwargs
) -> tp.List[Dataset]:
    num_workers: int = max(1, min(batch_size, max_workers))
    for n in range(max_workers, 1, -1):
        if batch_size % n == 0:
            num_workers = n
            break

    batch_size = batch_size // num_workers

    return [cls(batch_size=batch_size, **kwargs) for _ in range(num_workers)]


class MultiStreamDataLoader():
    def __init__(
        self,
        datasets: tp.List[Dataset],
        num_workers: int = 0,
    ):
        self.datasets = datasets
        self.num_workers = num_workers

    def get_stream_loaders(self):
        return zip(*[DataLoader(
            dataset, num_workers=self.num_workers, batch_size=None,
        ) for dataset in self.datasets])

    def __iter__(self):
        # TODO: optimize cast to tensor
        for batch_parts in self.get_stream_loaders():
            batch = list(itertools.chain(*batch_parts))
            batch_samples = torch.zeros((len(batch), 33 * 3))
            batch_labels = torch.zeros(len(batch), dtype=torch.long)
            for i, sample in enumerate(batch):
                batch_samples[i] = sample[0]
                batch_labels[i] = sample[1]
            yield batch_samples, batch_labels
