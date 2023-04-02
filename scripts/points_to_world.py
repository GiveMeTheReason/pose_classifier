import glob
import logging
import os
import sys

import imageio.v3 as iio
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

CAMERA = 'center'


def main():
    logger.info('Starting points to world script...')

    depth_base_path = os.path.join(
        CONFIG.path.undistorted,
    )
    file_paths = sorted(glob.glob(os.path.join(
        CONFIG.path.mediapipe_points,
        '*.csv',
    )))

    logger.info(f'Found {len(file_paths)} files')

    with open('scripts/csv_header.txt') as file:
        csv_header = [line.strip() for line in file]
    csv_header_str = ','.join(csv_header)

    image_size, camera_intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)
    intrinsic = utils.camera_params_to_ndarray(camera_intrinsic)

    for counter, file_path in enumerate(file_paths, start=1):
        logger.info(f'Start processing {counter}/{len(file_paths)} file: {file_path}')

        save_path = os.path.join(CONFIG.path.world_mp_points, os.path.basename(file_path))
        if os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        depth_paths = sorted(glob.glob(os.path.join(depth_base_path, *trial_info, f'cam_{CAMERA}', 'depth', '*.png')))

        mp_points = utils.get_mediapipe_points(file_path)

        for points, depth_path in zip(mp_points, depth_paths):
            depth_image = iio.imread(depth_path)
            frame_points = points.reshape(-1, 3)

            valid = utils.points_in_screen(frame_points)
            frame_points[~valid] = 0
            utils.screen_to_world(frame_points, depth_image, intrinsic, True)

        np.savetxt(save_path, mp_points, delimiter=',', header=csv_header_str, comments='')

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
