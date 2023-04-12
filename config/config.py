import os
import typing as tp

from config.base_config import ConfigBaseClass

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
points_pose_world_filtered = os.path.join(
    'mediapipe_data',
    'pose_world_filtered',
)
points_pose_world_windowed = os.path.join(
    'mediapipe_data',
    'pose_world_windowed',
)
points_pose_world_windowed_filtered = os.path.join(
    'mediapipe_data',
    'pose_world_windowed_filtered',
)
points_pose_world_windowed_filtered_labeled = os.path.join(
    'mediapipe_data',
    'pose_world_windowed_filtered_labeled',
)
labels = os.path.join(
    'mediapipe_data',
    'labels',
)

columns_pose = os.path.join(
    'scripts',
    'columns_pose.txt',
)

class Mediapipe(ConfigBaseClass):
    points_pose_raw: str = points_pose_raw
    points_pose_world: str = points_pose_world
    points_pose_world_filtered: str = points_pose_world_filtered
    points_pose_world_windowed: str = points_pose_world_windowed
    points_pose_world_windowed_filtered: str = points_pose_world_windowed_filtered
    points_pose_world_windowed_filtered_labeled: str = points_pose_world_windowed_filtered_labeled
    labels: str = labels
    columns_pose: str = columns_pose

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

output_data = os.path.join(
    'output_data',
)
use_wandb = True

train_share = 0.8

batch_size = 128 * 5
max_workers = 8

epochs = 10
validate_each_epoch = 1

learning_rate = 1e-4
weight_decay = 1e-5
weight_loss = [1.0] * len(gestures)
if with_rejection:
    weight_loss.append(1.0)

class TrainParams(ConfigBaseClass):
    seed: int = seed

    output_data: str = output_data
    use_wandb: bool = use_wandb

    train_share: float = train_share

    batch_size: int = batch_size
    max_workers: int = max_workers

    epochs: int = epochs
    validate_each_epoch: int = validate_each_epoch

    learning_rate: float = learning_rate
    weight_decay: float = weight_decay
    weight_loss: tp.List[float] = weight_loss

##################################################

class CONFIG(ConfigBaseClass):
    dataset: Dataset = Dataset()
    cameras: Cameras = Cameras()
    mediapipe: Mediapipe = Mediapipe()
    gesture_set: GestureSet = GestureSet()
    train_params: TrainParams = TrainParams()
