import dataclasses
import typing as tp


ConfigItemT = tp.TypeVar('ConfigItemT', bound='tp.Any')

@dataclasses.dataclass
class ConfigBaseClass:
    def __getitem__(self, item: ConfigItemT) -> ConfigItemT:
        return getattr(self, item)
