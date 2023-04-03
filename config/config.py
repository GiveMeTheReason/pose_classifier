import os
import typing as tp

from .base_config import ConfigBaseClass

##################################################

undistorted = os.path.join(
    'data',
    'undistorted',
)
sample_data = os.path.join(
    'data',
    'sample_data',
)
mediapipe_points = os.path.join(
    'data',
    'data-parsed',
)

class Dataset(ConfigBaseClass):
    undistorted: str = undistorted
    sample_data: str = sample_data
    mediapipe_points: str = mediapipe_points

##################################################

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

class Cameras(ConfigBaseClass):
    center_camera_params: str = center_camera_params
    left_camera_params: str = left_camera_params
    right_camera_params: str = right_camera_params

##################################################

points_pose_raw = os.path.join(
    'mediapipe_data',
    'pose_raw',
)
points_pose_world = os.path.join(
    'mediapipe_data',
    'pose_world',
)
csv_header_pose = os.path.join(
    'scripts',
    'csv_header.txt',
)

class Mediapipe(ConfigBaseClass):
    points_pose_raw: str = points_pose_raw
    points_pose_world: str = points_pose_world
    csv_header_pose: str = csv_header_pose

##################################################

gestures = (
    'select',
    'call',
    'start',
    'yes',
    'no',
)
with_rejection = True

class GestureSet(ConfigBaseClass):
    gestures: tp.Tuple[str, ...] = gestures
    with_rejection: bool = with_rejection

##################################################

seed = 0

class TrainParams(ConfigBaseClass):
    seed: int = seed

##################################################

class CONFIG(ConfigBaseClass):
    dataset: Dataset = Dataset()
    cameras: Cameras = Cameras()
    mediapipe: Mediapipe = Mediapipe()
    gesture_set: GestureSet = GestureSet()
    train_params: TrainParams = TrainParams()
