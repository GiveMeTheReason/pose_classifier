import itertools


POINTS_POSE_LIST = [
    'NOSE',                 # 0 (-)
    'NECK',                 # 1 (-)
    'RIGHT_SHOULDER',       # 2 (0)
    'RIGHT_ELBOW',          # 3 (1)
    'RIGHT_WRIST',          # 4 (2)
    'LEFT_SHOULDER',        # 5 (3)
    'LEFT_ELBOW',           # 6 (4)
    'LEFT_WRIST',           # 7 (5)
    'RIGHT_HIP',            # 8 (6)
    'RIGHT_KNEE',           # 9 (-)
    'RIGHT_ANKLE',          # 10 (-)
    'LEFT_HIP',             # 11 (7)
    'LEFT_KNEE',            # 12 (-)
    'LEFT_ANKLE',           # 13 (-)
    'RIGHT_EYE',            # 14 (-)
    'LEFT_EYE',             # 15 (-)
    'RIGHT_EAR',            # 16 (-)
    'LEFT_EAR',             # 17 (-)
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
