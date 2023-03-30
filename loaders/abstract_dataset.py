import abc
import itertools
import random
import typing as tp

from torch.utils.data import IterableDataset


class AbstractDataset(abc.ABC, IterableDataset):

    @abc.abstractmethod
    def __init__(
        self,
        samples: tp.Sequence[tp.Any],
        label_map: tp.Dict[tp.Hashable, int],
        batch_size: int = 1,
        transforms: tp.Any = None,
        *args,
        **kwargs,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.batch_size = batch_size
        self.transforms = transforms

    @abc.abstractmethod
    def _extract_label(self, sample: tp.Any) -> int:
        pass

    @abc.abstractmethod
    def _transform_sample(self, sample: tp.Any) -> tp.Any:
        pass

    @abc.abstractmethod
    def process_samples(
        self,
        sample: tp.Any,
        batch_idx: int,
    ):
        pass

    @property
    def shuffle_samples(self) -> tp.List[tp.Any]:
        return random.sample(self.samples, len(self.samples))

    def get_stream(self, data_list: tp.List[str], batch_idx: int):
        return itertools.chain.from_iterable(
            map(self.process_samples, data_list, itertools.repeat(batch_idx))
        )

    def get_streams(self):
        return zip(*[self.get_stream(
            self.shuffle_samples, batch_idx
        ) for batch_idx in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()
