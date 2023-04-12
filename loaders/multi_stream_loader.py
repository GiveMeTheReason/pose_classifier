import itertools
import typing as tp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MultiStreamDataLoader():
    def __init__(
        self,
        datasets: tp.List[Dataset],
        num_workers: int = 0,
    ) -> None:
        self.datasets = datasets
        self.num_workers = num_workers

    def get_stream_loaders(self) -> tp.Iterator[tp.Tuple[DataLoader]]:
        return zip(*[DataLoader(
            dataset, num_workers=self.num_workers, batch_size=None,
        ) for dataset in self.datasets])

    def __iter__(self) -> tp.Generator[tp.Tuple[torch.Tensor, torch.Tensor], None, None]:
        for batch_parts in self.get_stream_loaders():
            batch = list(itertools.chain(*batch_parts))
            batch_samples = torch.stack([item[0] for item in batch])
            batch_labels = torch.stack([item[1] for item in batch])
            yield batch_samples, batch_labels
