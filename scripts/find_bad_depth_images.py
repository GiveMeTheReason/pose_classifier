import glob
import logging
import os
import sys

import imageio.v3 as iio

from config import CONFIG


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
    depth_base_path = os.path.join(
        CONFIG.dataset.undistorted,
    )
    file_paths = sorted(glob.glob(os.path.join(
        CONFIG.mediapipe.points_pose_raw,
        '*.npy',
    )))

    err_depth_images = []
    logger.info(f'Found {len(file_paths)} files')

    for counter, file_path in enumerate(file_paths, start=1):
        logger.info(f'Start processing {counter}/{len(file_paths)} file: {file_path}')

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        trial_info[1] = trial_info[1].replace('-', '_')
        depth_paths = sorted(glob.glob(os.path.join(depth_base_path, *trial_info, f'cam_{CAMERA}', 'depth', '*.png')))

        for depth_path in depth_paths:
            depth_image = iio.imread(depth_path)
            if depth_image.max() == 0:
                err_depth_images.append(depth_path)
                logger.error(depth_path)


    for err in err_depth_images:    
        logger.error(err)


if __name__ == '__main__':
    main()
