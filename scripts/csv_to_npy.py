import glob
import os
import tqdm

import numpy as np

import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
from config import DATA_CONFIG


logger = utils_logging.init_logger(__name__)

FOLDER = DATA_CONFIG.mediapipe.points_pose_world
FORCE = False


def main():
    logger.info(f'Starting csv to npy script for folder: {FOLDER}')

    file_paths = glob.glob(os.path.join(
        FOLDER,
        '*.csv',
    ))

    logger.info(f'Found {len(file_paths)} files')

    counter = 0
    for file_path in tqdm.tqdm(file_paths):
        save_path = os.path.join(
            FOLDER,
            os.path.splitext(os.path.basename(file_path))[0] + '.npy',
        )

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        file_data = utils_mediapipe.load_points(file_path, skip_header=1)
        np.save(save_path, file_data, fix_imports=False)

        counter += 1

    logger.info(f'Converted totally: {counter} files')


if __name__ == '__main__':
    main()
