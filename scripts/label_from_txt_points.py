import glob
import os
import tqdm

import numpy as np

import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
from config import DATA_CONFIG


logger = utils_logging.init_logger(__name__)

FORCE = False

POINTS_FOLDER = DATA_CONFIG.mediapipe.points_unified_world_filtered
SAVE_FOLDER = DATA_CONFIG.mediapipe.points_unified_world_filtered_labeled
LABELS_TXT_FOLDER = DATA_CONFIG.mediapipe.labels


def filename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    logger.info('Starting labeling points from txt script...')

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    file_paths = glob.glob(os.path.join(
        POINTS_FOLDER,
        '*.npy',
    ))

    logger.info(f'Found {len(file_paths)} files')

    for file_path in tqdm.tqdm(file_paths):
        label_path = os.path.join(
            LABELS_TXT_FOLDER,
            f'{filename_without_ext(file_path)}.txt',
        )
        if not os.path.exists(label_path):
            logger.error(f'Files mismatch: {file_path} and {label_path}')
            continue

        save_path = os.path.join(SAVE_FOLDER, os.path.basename(file_path))

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        mp_points = utils_mediapipe.load_points(file_path)
        labels = utils_mediapipe.load_points(label_path).astype(int)
        labeled_points = np.append(mp_points, np.zeros((mp_points.shape[0], 1)), axis=1)

        gesture_start = labels[0] - 1
        gesture_end = labels[1]

        labeled_points[gesture_start+1:gesture_end, -1] = 1

        np.save(save_path, labeled_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
