import dataclasses
import os

##################################################

mediapipe_points = os.path.join(
    'data',
    'data-parsed',
)
undistorted = os.path.join(
    'data',
    'undistorted',
)

@dataclasses.dataclass
class PATH:
    mediapipe_points: str = mediapipe_points
    undistorted: str = undistorted

##################################################

@dataclasses.dataclass
class CONFIG:
    path: PATH = PATH()
