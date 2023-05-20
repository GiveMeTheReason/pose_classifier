import dataclasses
import os
import typing as tp

import numpy as np


POINTS_LIST = [
    'NOSE',                 # 0 (-)
    'LEFT_EYE_INNER',       # 1 (-)
    'LEFT_EYE',             # 2 (-)
    'LEFT_EYE_OUTER',       # 3 (-)
    'RIGHT_EYE_INNER',      # 4 (-)
    'RIGHT_EYE',            # 5 (-)
    'RIGHT_EYE_OUTER',      # 6 (-)
    'LEFT_EAR',             # 7 (-)
    'RIGHT_EAR',            # 8 (-)
    'MOUTH_LEFT',           # 9 (-)
    'MOUTH_RIGHT',          # 10 (-)
    'LEFT_SHOULDER',        # 11 (0)
    'RIGHT_SHOULDER',       # 12 (1)
    'LEFT_ELBOW',           # 13 (2)
    'RIGHT_ELBOW',          # 14 (3)
    'LEFT_WRIST',           # 15 (4)
    'RIGHT_WRIST',          # 16 (5)
    'LEFT_PINKY',           # 17 (6)
    'RIGHT_PINKY',          # 18 (7)
    'LEFT_INDEX',           # 19 (8)
    'RIGHT_INDEX',          # 20 (9)
    'LEFT_THUMB',           # 21 (10)
    'RIGHT_THUMB',          # 22 (11)
    'LEFT_HIP',             # 23 (12)
    'RIGHT_HIP',            # 24 (13)
    'LEFT_KNEE',            # 25 (-)
    'RIGHT_KNEE',           # 26 (-)
    'LEFT_ANKLE',           # 27 (-)
    'RIGHT_ANKLE',          # 28 (-)
    'LEFT_HEEL',            # 29 (-)
    'RIGHT_HEEL',           # 30 (-)
    'LEFT_FOOT_INDEX',      # 31 (-)
    'RIGHT_FOOT_INDEX',     # 32 (-)
]
POINTS_MAP = {idx: name for idx, name in enumerate(POINTS_LIST)}


@dataclasses.dataclass
class Landmark:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0


EMPTY_POSE = (Landmark() for _ in range(33))
EMPTY_HAND = (Landmark() for _ in range(21))


def get_mediapipe_points(mp_points_path: str, **kwargs) -> np.ndarray:
    file_format = os.path.splitext(mp_points_path)[-1]
    if file_format == '.npy':
        return _get_mediapipe_points_npy(mp_points_path, **kwargs)
    if file_format == '.txt':
        return _get_mediapipe_points_txt(mp_points_path, **kwargs)
    if file_format == '.csv':
        return _get_mediapipe_points_txt(mp_points_path, delimiter=',', **kwargs)
    raise Exception(f'Unknown data format {mp_points_path}')


def _get_mediapipe_points_txt(mp_points_path: str, **kwargs) -> np.ndarray:
    return np.genfromtxt(mp_points_path, **kwargs)


def _get_mediapipe_points_npy(mp_points_path: str, **kwargs) -> np.ndarray:
    return np.load(mp_points_path, allow_pickle=True, **kwargs)


def get_points_from_image(solver, image: np.ndarray) -> np.ndarray:
    landmarks = solver.process(image)
    if landmarks.pose_landmarks is not None:
        frame_points = landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
        frame_points = frame_points.reshape(-1)
        return frame_points
    return np.zeros(0)


def landmarks_to_array(landmarks: tp.Iterable[Landmark]) -> np.ndarray:
    array = [
        [landmark.x, landmark.y, landmark.z, landmark.visibility]
        for landmark in landmarks
    ]
    return np.array(array)
