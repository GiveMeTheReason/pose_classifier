import glob
import logging
import os
import sys

import numpy as np


logger = logging.getLogger(__name__)
strfmt = '[%(asctime)s] [%(levelname)-5.5s] %(message)s'
datefmt = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=strfmt, datefmt=datefmt)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


BASE_FOLDER = os.path.join(
    'mediapipe_data',
    'pose_world',
)
FORCE = False


def main():
    logger.info(f'Starting csv to npy script for folder: {BASE_FOLDER}')

    file_paths = glob.iglob(os.path.join(
        BASE_FOLDER,
        '*.csv',
    ))

    counter = 0
    for file_path in file_paths:
        save_path = os.path.join(
            BASE_FOLDER,
            os.path.splitext(os.path.basename(file_path))[0] + '.npy'
        )

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        file_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        np.save(save_path, file_data, fix_imports=False)

        counter += 1

    logger.info(f'Converted totally: {counter} files')


if __name__ == '__main__':
    main()
