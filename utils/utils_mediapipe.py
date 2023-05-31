import dataclasses
import itertools
import os
import typing as tp

import numpy as np

import utils.utils_unified_format as utils_unified_format


POINTS_POSE_LIST = [
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
POINTS_HAND_LIST = [
    'WRIST',                # 0 (-)
    'THUMB_1',              # 1 (-)
    'THUMB_2',              # 2 (-)
    'THUMB_3',              # 3 (-)
    'THUMB_4',              # 4 (-)
    'INDEX_1',              # 5 (-)
    'INDEX_2',              # 6 (-)
    'INDEX_3',              # 7 (-)
    'INDEX_4',              # 8 (-)
    'MIDDLE_1',             # 9 (-)
    'MIDDLE_2',             # 10 (-)
    'MIDDLE_3',             # 11 (-)
    'MIDDLE_4',             # 12 (-)
    'RING_1',               # 13 (-)
    'RING_2',               # 14 (-)
    'RING_3',               # 15 (-)
    'RING_4',               # 16 (-)
    'PINKY_1',              # 17 (-)
    'PINKY_2',              # 18 (-)
    'PINKY_3',              # 19 (-)
    'PINKY_4',              # 20 (-)
]

POINTS_LEFT_HAND_LIST = [f'LEFT_{name}' for name in POINTS_HAND_LIST]
POINTS_RIGHT_HAND_LIST = [f'RIGHT_{name}' for name in POINTS_HAND_LIST]
POINTS_MAP = {idx: name for idx, name in enumerate(
    itertools.chain(POINTS_POSE_LIST, POINTS_LEFT_HAND_LIST, POINTS_RIGHT_HAND_LIST)
)}

POSE_POINTS_COUNT = len(POINTS_POSE_LIST)
HAND_POINTS_COUNT = len(POINTS_HAND_LIST)
TOTAL_POINTS_COUNT = POSE_POINTS_COUNT + 2 * HAND_POINTS_COUNT

POSE_SLICE = slice(POSE_POINTS_COUNT)
LEFT_HAND_SLICE = slice(POSE_POINTS_COUNT, POSE_POINTS_COUNT + HAND_POINTS_COUNT)
RIGHT_HAND_SLICE = slice(POSE_POINTS_COUNT + HAND_POINTS_COUNT, TOTAL_POINTS_COUNT)

MP_TO_UNIFIED_KEYS = [
    0,
    0,  # needs further transforms
    12,
    14,
    16,
    11,
    13,
    15,
    24,
    26,
    28,
    23,
    25,
    27,
    5,
    2,
    8,
    7,
]


@dataclasses.dataclass
class Landmark:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0


EMPTY_POSE = tuple(Landmark() for _ in range(POSE_POINTS_COUNT))
EMPTY_HAND = tuple(Landmark() for _ in range(HAND_POINTS_COUNT))


def load_points(mp_points_path: str, **kwargs) -> np.ndarray:
    file_format = os.path.splitext(mp_points_path)[-1]
    if file_format == '.npy':
        return _load_points_npy(mp_points_path, **kwargs)
    if file_format == '.txt':
        return _load_points_txt(mp_points_path, **kwargs)
    if file_format == '.csv':
        return _load_points_txt(mp_points_path, delimiter=',', **kwargs)
    raise Exception(f'Unknown data format {mp_points_path}')


def _load_points_txt(mp_points_path: str, **kwargs) -> np.ndarray:
    return np.genfromtxt(mp_points_path, **kwargs)


def _load_points_npy(mp_points_path: str, **kwargs) -> np.ndarray:
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


def mediapipe_to_unified(mp_points: np.ndarray) -> np.ndarray:
    unified_shape = [*mp_points.shape]
    unified_shape[-2] = utils_unified_format.TOTAL_POINTS_COUNT

    unified_array = np.zeros(unified_shape)

    # pose
    unified_array[..., range(utils_unified_format.POSE_POINTS_COUNT), :] = mp_points[..., MP_TO_UNIFIED_KEYS, :]
    unified_array[..., 1, :] = np.mean(mp_points[..., [11, 12], :], axis=-2)

    # hands
    unified_array[..., utils_unified_format.POSE_POINTS_COUNT:, :] = mp_points[..., POSE_POINTS_COUNT:, :]

    return unified_array
