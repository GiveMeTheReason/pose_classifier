import os
import typing as tp

from config.base_config import ConfigBaseClass

##################################################

class SelectRight(ConfigBaseClass):
    mp_point: str = 16
    bound_percentile: tp.Tuple[float, float] = (0.1, 0.9)
    threshold: str = 0.15

class SelectLeft(ConfigBaseClass):
    mp_point: str = 15
    bound_percentile: tp.Tuple[float, float] = (0.1, 0.9)
    threshold: str = 0.15

##################################################

##################################################

class LABEL_CONFIG(ConfigBaseClass):
    select_right: SelectRight = SelectRight()

