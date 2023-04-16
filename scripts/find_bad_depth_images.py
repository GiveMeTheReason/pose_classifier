import glob
import os
import tqdm

import imageio.v3 as iio

import utils.utils_logging as utils_logging
from config import DATA_CONFIG


logger = utils_logging.init_logger(__name__)

DEPTH_FOLDER = DATA_CONFIG.dataset.undistorted
ANALYSE_FOLDER = DATA_CONFIG.mediapipe.points_pose_raw
CAMERA = 'center'


def main():
    logger.info('Starting finding corrupted depth images script...')

    file_paths = glob.glob(os.path.join(
        ANALYSE_FOLDER,
        '*.npy',
    ))

    logger.info(f'Found {len(file_paths)} files')

    for file_path in tqdm.tqdm(file_paths):
        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        trial_info[1] = trial_info[1].replace('-', '_')
        depth_paths = glob.iglob(os.path.join(
            DEPTH_FOLDER,
            *trial_info,
            f'cam_{CAMERA}',
            'depth',
            '*.png',
        ))

        for depth_path in depth_paths:
            depth_image = iio.imread(depth_path).T
            if depth_image.max() == 0:
                logger.error(f'Corrupted depth image: {depth_path}')


if __name__ == '__main__':
    main()
