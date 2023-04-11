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

CAMERA = 'center'
FORCE = True
WINDOWED = True
WITH_XY = True

if WINDOWED:
    base_points_path = CONFIG.mediapipe.points_pose_world_windowed
    base_save_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered
else:
    base_points_path = CONFIG.mediapipe.points_pose_world
    base_save_folder = CONFIG.mediapipe.points_pose_world_filtered

DELTA_T = 1 / 30
SIGMA_U = 20 * 500
SIGMA_Z = 20
F = [
    [1, DELTA_T],
    [0, 1],
]
H = [
    [1, 0],
]
P = [
    [20 ** 2, 0],
    [0, 0.1 ** 2],
]
R = [[1]]
Q = [
    [1/4 * DELTA_T ** 2, 1/2 * DELTA_T],
    [1/2 * DELTA_T, 1],
]


def init_filter(init_value: float = None) -> KalmanFilter:
    kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=0)

    kf.x = np.array([
        [init_value or 0.0],
        [0.0],
    ])
    kf.F = np.array(F)
    kf.H = np.array(H)
    kf.P = np.array(P)
    kf.R = np.array(R) * (SIGMA_Z ** 2)
    kf.Q = np.array(Q) * (DELTA_T ** 2) * (SIGMA_U ** 2)
    return kf


def md_sigma(md: float) -> float:
    return 1 + 1 * md ** 2


def main():
    logger.info('Starting points to filtered script...')

    file_paths = sorted(glob.glob(os.path.join(
        base_points_path,
        '*.npy',
    )))

    if not os.path.exists(base_save_folder):
        os.makedirs(base_save_folder, exist_ok=True)

    image_size, intrinsic = utils.get_camera_params(CONFIG.cameras[f'{CAMERA}_camera_params'])
    width, height = image_size
    focal_x = intrinsic[0, 0]
    focal_y = intrinsic[1, 1]
    principal_x = intrinsic[0, 2]
    principal_y = intrinsic[1, 2]

    logger.info(f'Found {len(file_paths)} files')

    for counter, file_path in enumerate(file_paths, start=1):
        logger.info(f'Start processing {counter}/{len(file_paths)} file: {file_path}')

        raw_file_path = file_path.replace(base_points_path, CONFIG.mediapipe.points_pose_raw)

        save_path = os.path.join(base_save_folder, os.path.basename(file_path))
        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        raw_points = utils.get_mediapipe_points(raw_file_path)
        mp_points = utils.get_mediapipe_points(file_path)
        assert raw_points.shape == mp_points.shape
        filtered_points = np.zeros_like(mp_points)
        filtered_points[0, 2::3] = mp_points[0, 2::3]
        filtered_points[0, 0::3] = (raw_points[0, 0::3] * width - principal_x) / focal_x * filtered_points[0, 2::3]
        filtered_points[0, 1::3] = (raw_points[0, 1::3] * height - principal_y) / focal_y * filtered_points[0, 2::3]
        kalman_filters = [init_filter(value) for value in mp_points[0]]

        for i, points in enumerate(mp_points[1:], start=1):
            for j, new_point in enumerate(points[2::3]):
                kf_z = kalman_filters[3*j+2]
                kf_z.predict()

                z_res = kf_z.residual_of(new_point)[0, 0]
                md = np.sqrt((z_res * z_res) / kf_z.P[0, 0])
                kf_z.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

                if new_point > 50:
                    kf_z.update(new_point)

                filtered_points[i, 3*j+2] = kf_z.x[0, 0]

                if WITH_XY:
                    new_x = (raw_points[i, 3*j+0] * width - principal_x) / focal_x * filtered_points[i, 3*j+2]
                    new_y = (raw_points[i, 3*j+1] * height - principal_y) / focal_y * filtered_points[i, 3*j+2]

                    kf_x = kalman_filters[3*j+0]
                    kf_y = kalman_filters[3*j+1]
                    kf_x.predict()
                    kf_y.predict()

                    x_res = kf_x.residual_of(new_x)[0, 0]
                    md = np.sqrt((x_res * x_res) / kf_x.P[0, 0])
                    kf_x.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

                    y_res = kf_y.residual_of(new_y)[0, 0]
                    md = np.sqrt((y_res * y_res) / kf_y.P[0, 0])
                    kf_y.R = np.array(R) * ((md_sigma(md) * SIGMA_Z) ** 2)

                    kf_x.update(new_x)
                    kf_y.update(new_y)

                    filtered_points[i, 3*j+0] = kf_x.x[0, 0]
                    filtered_points[i, 3*j+1] = kf_y.x[0, 0]

        if not WITH_XY:
            filtered_points[:, 0::3] = (raw_points[:, 0::3] * width - principal_x) / focal_x * filtered_points[:, 2::3]
            filtered_points[:, 1::3] = (raw_points[:, 1::3] * height - principal_y) / focal_y * filtered_points[:, 2::3]

        np.save(save_path, filtered_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
