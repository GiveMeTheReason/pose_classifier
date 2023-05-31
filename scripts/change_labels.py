import os

import numpy as np

import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
from config import DATA_CONFIG


logger = utils_logging.init_logger(__name__)

SAVE_FOLDER = DATA_CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled
LABELS_SAVE_FOLDER = DATA_CONFIG.mediapipe.labels

SUBJECT = 120
GESTURE = 'start'
HAND = 'right'
TRIAL = 1

LABEL_START = 28
LABEL_END = 47


def main():
    logger.info('Starting labeling change script...')

    file_path = os.path.join(
        SAVE_FOLDER,
        f'G{SUBJECT}_{GESTURE}_{HAND}_trial{TRIAL}.npy',
    )
    labels_save_path = os.path.join(
        LABELS_SAVE_FOLDER,
        os.path.basename(file_path).replace('.npy', '.txt'),
    )

    labeled_points = utils_mediapipe.load_points(file_path)
    labeled_points[:, -1] = 0
    labeled_points[LABEL_START+1:LABEL_END, -1] = 1

    np.save(file_path, labeled_points, fix_imports=False)

    with open(labels_save_path, 'w') as labels_file:
        labels_file.write(f'{LABEL_START} {LABEL_END}')

    logger.info(f'Saved at: {file_path}')


if __name__ == '__main__':
    main()
