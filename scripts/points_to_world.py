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
FORCE = False
WINDOWED = True


def main():
    logger.info('Starting points to world script...')

    depth_base_path = os.path.join(
        CONFIG.dataset.undistorted,
    )
    file_paths = sorted(glob.glob(os.path.join(
        CONFIG.mediapipe.points_pose_raw,
        '*.npy',
    )))

    logger.info(f'Found {len(file_paths)} files')

    with open(CONFIG.mediapipe.columns_pose) as file:
        csv_header = [line.strip() for line in file]
    csv_header_str = ','.join(csv_header)

    image_size, intrinsic = utils.get_camera_params(CONFIG.cameras[f'{CAMERA}_camera_params'])

    depth_extractor = utils.DepthExtractor(*image_size, intrinsic)

    err_depth_images = []

    for counter, file_path in enumerate(file_paths, start=1):
        logger.info(f'Start processing {counter}/{len(file_paths)} file: {file_path}')

        save_path = os.path.join(CONFIG.mediapipe.points_pose_world + WINDOWED * '_windowed', os.path.basename(file_path))
        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        trial_info[1] = trial_info[1].replace('-', '_')
        depth_paths = sorted(glob.glob(os.path.join(depth_base_path, *trial_info, f'cam_{CAMERA}', 'depth', '*.png')))
        last_stable_depth = depth_paths[0]

        mp_points = utils.get_mediapipe_points(file_path)

        depth_extractor.is_inited = False

        if len(mp_points) != len(depth_paths):
            logger.error(
                f'Lenght of points ({len(mp_points)}) and '
                f'depth images ({len(depth_paths)}) does not match!'
            )
            continue

        for points, depth_path in zip(mp_points, depth_paths):
            depth_image = iio.imread(depth_path)
            if depth_image.max() == 0:
                err_depth_images.append(depth_path)
                depth_image = iio.imread(last_stable_depth)
            last_stable_depth = depth_path
            frame_points = points.reshape(-1, 3)

            valid = utils.points_in_screen(frame_points)
            frame_points[~valid] = 0
            depth_extractor.screen_to_world(frame_points, depth_image, WINDOWED, True)

        np.save(save_path, mp_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')

    for err in err_depth_images:    
        logger.error(err)


if __name__ == '__main__':
    main()
