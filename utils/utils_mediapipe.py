import dataclasses
import os
import typing as tp

import numpy as np


@dataclasses.dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


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


def landmarks_to_array(landmarks: tp.Sequence[Landmark]) -> np.ndarray:
    array = [
        [landmark.x, landmark.y, landmark.z, landmark.visibility]
        for landmark in landmarks
    ]
    return np.array(array)
