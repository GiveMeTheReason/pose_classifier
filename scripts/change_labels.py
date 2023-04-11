import glob
import logging
import os
import sys
import typing as tp

import numpy as np
from filterpy.kalman import KalmanFilter

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

base_save_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled
base_labels_save_folder = CONFIG.mediapipe.labels

SUBJECT = 120
GESTURE = 'start'
HAND = 'right'
TRIAL = 1

gesture_start = 28
gesture_end = 47


def main():
    logger.info('Starting labeling change script...')

    file_path = os.path.join(
        base_save_folder,
        f'G{SUBJECT}_{GESTURE}_{HAND}_trial{TRIAL}.npy',
    )
    labels_save_path = os.path.join(base_labels_save_folder, os.path.basename(file_path).replace('.npy', '.txt'))

    labeled_points = utils.get_mediapipe_points(file_path)
    labeled_points[:, -1] = 0
    labeled_points[gesture_start+1:gesture_end, -1] = 1

    np.save(file_path, labeled_points, fix_imports=False)

    with open(labels_save_path, 'w') as labels_file:
        labels_file.write(f'{gesture_start} {gesture_end}')

    logger.info(f'Saved at: {file_path}')


if __name__ == '__main__':
    main()
