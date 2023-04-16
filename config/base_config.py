import dataclasses
import typing as tp


ConfigItemT = tp.TypeVar('ConfigItemT', bound='tp.Any')

@dataclasses.dataclass
class ConfigBaseClass:
    def __getitem__(self, item: ConfigItemT) -> ConfigItemT:
        return getattr(self, item)

    @classmethod
    def as_dict(cls) -> tp.Dict[str, tp.Any]:
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith('__')
        }
