import glob
import os

import numpy as np

from config import CONFIG
import visualizer.utils as utils


def main():
    # [101, 120]
    SUBJECT = 101
    # ...
    GESTURE = 'select'
    # ['both', 'left', 'right']
    HAND = 'left'
    # [1, 4...6]
    TRIAL = 1
    # ['center', 'left', 'right']
    CAMERA = 'center'
    # [0, 120]
    FRAME = 0

    folder_path = os.path.join(
        CONFIG.path.undistorted,
        f'G{str(SUBJECT).zfill(3)}',
        GESTURE,
        HAND,
        f'trial{TRIAL}',
        f'cam_{CAMERA}',
    )

    color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))

    color_image_path = color_paths[FRAME]
    depth_image_path = depth_paths[FRAME]

    rgbd_image = utils.get_rgbd_image(
        color_image_path,
        depth_image_path,
    )

    image_size, camera_intrinsic = utils.get_camera_params(CONFIG.path.camera_params)
    intrinsic = utils.camera_params_to_ndarray(camera_intrinsic)

    point_cloud = utils.create_point_cloud(
        rgbd_image,
        image_size,
        intrinsic=intrinsic,
        extrinsic=np.eye(4),
    )

    utils.visualize_point_cloud(
        np.asarray(point_cloud.points),
        np.asarray(point_cloud.colors),
        step=100,
        with_axis=True,
    )


if __name__ == '__main__':
    main()
