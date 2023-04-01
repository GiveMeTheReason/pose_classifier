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
camera_params = os.path.join(
    'config',
    'camera_params.json',
)

@dataclasses.dataclass
class PATH:
    mediapipe_points: str = mediapipe_points
    undistorted: str = undistorted
    camera_params: str = camera_params

##################################################



##################################################

@dataclasses.dataclass
class CONFIG:
    path: PATH = PATH()
