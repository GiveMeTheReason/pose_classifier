import glob
import os
import tqdm

import numpy as np

import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
from config import DATA_CONFIG


logger = utils_logging.init_logger(__name__)

FORCE = False

POINTS_FOLDER = DATA_CONFIG.mediapipe.points_pose_world_windowed_filtered
SAVE_FOLDER = DATA_CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled
LABELS_SAVE_FOLDER = DATA_CONFIG.mediapipe.labels


def main():
    logger.info('Starting labeling points script...')

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)
    if not os.path.exists(LABELS_SAVE_FOLDER):
        os.makedirs(LABELS_SAVE_FOLDER, exist_ok=True)

    file_paths = glob.glob(os.path.join(
        POINTS_FOLDER,
        '*.npy',
    ))

    logger.info(f'Found {len(file_paths)} files')

    for file_path in tqdm.tqdm(file_paths):
        save_path = os.path.join(SAVE_FOLDER, os.path.basename(file_path))
        labels_save_path = os.path.join(LABELS_SAVE_FOLDER, os.path.basename(file_path).replace('.npy', '.txt'))

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        subject, gesture, hand, trial = trial_info

        if hand == 'left':
            target_point = 15
        else:
            target_point = 16
 
        mp_points = utils_mediapipe.get_mediapipe_points(file_path)
        labeled_points = np.append(mp_points, np.zeros((mp_points.shape[0], 1)), axis=1)

        screening_points = mp_points[:, 3*target_point+1]

        lower_bound = np.quantile(screening_points, 0.1)
        upper_bound = np.quantile(screening_points, 0.9)
        threshold = 0.1 * (upper_bound - lower_bound) + lower_bound

        gesture_start = np.argmax(screening_points < threshold)
        gesture_end = screening_points.shape[0] - np.argmax(screening_points[::-1] < threshold)

        labeled_points[gesture_start+1:gesture_end, -1] = 1

        np.save(save_path, labeled_points, fix_imports=False)
        with open(labels_save_path, 'w') as labels_file:
            labels_file.write(f'{gesture_start} {gesture_end}')

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
