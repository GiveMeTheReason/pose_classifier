import glob
import logging
import os
import sys

import numpy as np

from config import CONFIG
import visualizer.utils as utils


logger = logging.getLogger(__name__)
strfmt = '[%(asctime)s] [%(levelname)-5.5s] %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=strfmt, datefmt=datefmt)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

FORCE = False

base_points_path = CONFIG.mediapipe.points_pose_world_windowed_filtered
base_save_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled
base_labels_save_folder = CONFIG.mediapipe.labels


def main():
    logger.info('Starting labeling points script...')

    file_paths = sorted(glob.glob(os.path.join(
        base_points_path,
        '*.npy',
    )))

    if not os.path.exists(base_save_folder):
        os.makedirs(base_save_folder, exist_ok=True)
    if not os.path.exists(base_labels_save_folder):
        os.makedirs(base_labels_save_folder, exist_ok=True)

    logger.info(f'Found {len(file_paths)} files')

    for counter, file_path in enumerate(file_paths, start=1):
        logger.info(f'Start processing {counter}/{len(file_paths)} file: {file_path}')

        save_path = os.path.join(base_save_folder, os.path.basename(file_path))
        labels_save_path = os.path.join(base_labels_save_folder, os.path.basename(file_path).replace('.npy', '.txt'))
        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        subject, gesture, hand, trial = trial_info

        if hand == 'left':
            target_point = 15
        else:
            target_point = 16
 
        mp_points = utils.get_mediapipe_points(file_path)
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
