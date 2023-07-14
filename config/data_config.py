import os

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

points_holistic_raw = os.path.join(
    'mediapipe_data',
    'holistic_raw',
)
points_holistic_world = os.path.join(
    'mediapipe_data',
    'holistic_world',
)

points_unified_world_filtered = os.path.join(
    'mediapipe_data',
    'unified_world_filtered',
)
points_unified_world_filtered_labeled = os.path.join(
    'mediapipe_data',
    'unified_world_filtered_labeled',
)

labels = os.path.join(
    'mediapipe_data',
    'labels',
)

columns_pose = os.path.join(
    'config',
    'columns_pose.txt',
)

class Mediapipe(ConfigBaseClass):
    points_pose_raw: str = points_pose_raw
    points_pose_world: str = points_pose_world
    points_pose_world_filtered: str = points_pose_world_filtered
    points_pose_world_windowed: str = points_pose_world_windowed
    points_pose_world_windowed_filtered: str = points_pose_world_windowed_filtered
    points_pose_world_windowed_filtered_labeled: str = points_pose_world_windowed_filtered_labeled

    points_holistic_raw: str = points_holistic_raw
    points_holistic_world: str = points_holistic_world
    points_unified_world_filtered: str = points_unified_world_filtered
    points_unified_world_filtered_labeled: str = points_unified_world_filtered_labeled

    labels: str = labels
    columns_pose: str = columns_pose

##################################################

stream_1 = os.path.join(
    'stream_data',
    'recording_001',
)

class StreamingDataset(ConfigBaseClass):
    stream_1: str = stream_1

##################################################

class DATA_CONFIG(ConfigBaseClass):
    dataset: Dataset = Dataset()
    cameras: Cameras = Cameras()
    mediapipe: Mediapipe = Mediapipe()
    streaming: StreamingDataset = StreamingDataset()
