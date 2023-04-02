import dataclasses
import os

##################################################

mediapipe_points = os.path.join(
    'saved_data',
)
world_mp_points = os.path.join(
    'saved_data',
    'world',
)
undistorted = os.path.join(
    'data',
    'undistorted',
)
sample_data = os.path.join(
    'data',
    'sample_data',
)
center_camera_params = os.path.join(
    'config',
    'center_camera_params.json',
)
left_camera_params = os.path.join(
    'config',
    'left_camera_params.json',
)
right_camera_params = os.path.join(
    'config',
    'right_camera_params.json',
)
csv_header = os.path.join(
    'scripts',
    'csv_header.txt',
)

@dataclasses.dataclass
class PATH:
    mediapipe_points: str = mediapipe_points
    world_mp_points: str = world_mp_points
    undistorted: str = undistorted
    sample_data: str = sample_data
    center_camera_params: str = center_camera_params
    left_camera_params: str = left_camera_params
    right_camera_params: str = right_camera_params
    csv_header: str = csv_header

##################################################



##################################################

@dataclasses.dataclass
class CONFIG:
    path: PATH = PATH()
