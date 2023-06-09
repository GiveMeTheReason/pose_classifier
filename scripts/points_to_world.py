import glob
import os
import typing as tp
import tqdm

import imageio.v3 as iio
import numpy as np

import utils.utils_camera_systems as utils_camera_systems
import utils.utils_kalman_filter as utils_kalman_filter
import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
import utils.utils_unified_format as utils_unified_format
from config import DATA_CONFIG, KALMAN_FILTER_CONFIG


logger = utils_logging.init_logger(__name__)

NEED_FILTERING = True
WINDOW_SIZE = 7

DEPTH_FOLDER = DATA_CONFIG.dataset.undistorted
RAW_POINTS_FOLDER = DATA_CONFIG.mediapipe.points_holistic_raw
SAVE_FOLDER = DATA_CONFIG.mediapipe.points_unified_world_filtered

KALMAN_PARAMS = KALMAN_FILTER_CONFIG.init_params.as_dict()
KALMAN_HEURISTICS_FUNC = KALMAN_FILTER_CONFIG.heuristics.as_dict()

CAMERA = 'center'
CAMERA_PARAMS_PATH = DATA_CONFIG.cameras[f'{CAMERA}_camera_params']
FORCE = False


def main():
    logger.info('Starting points to world script...')

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(
        RAW_POINTS_FOLDER,
        '*.npy',
        # 'G101_select_left_trial2.npy',
    )))

    logger.info(f'Found {len(file_paths)} files')

    image_size, intrinsic = utils_camera_systems.get_camera_params(CAMERA_PARAMS_PATH)
    camera_systems = utils_camera_systems.CameraSystems(image_size, intrinsic)
    depth_extractor = utils_camera_systems.DepthExtractor(WINDOW_SIZE)

    kfs = []
    for i in range(utils_unified_format.TOTAL_POINTS_COUNT):
        point = i
        if point >= 18:
            point = 4
        params = KALMAN_FILTER_CONFIG.init_params.as_dict()
        params['sigma_u'] = params.pop('sigma_u_points')[point]
        params['init_Q'] = np.copy(params['init_Q']) * (params['sigma_u'] ** 2)
        kfs.append(utils_kalman_filter.KalmanFilter(**params, **KALMAN_HEURISTICS_FUNC))
    kalman_filters = utils_kalman_filter.KalmanFilters(kfs)

    # val_acc = []
    # val_sigma_acc = []

    for file_path in tqdm.tqdm(file_paths):
        save_path = os.path.join(SAVE_FOLDER, os.path.basename(file_path))

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        trial_info = os.path.splitext(os.path.basename(file_path))[0].split('_')
        trial_info[1] = trial_info[1].replace('-', '_')
        depth_paths = sorted(glob.glob(os.path.join(
            DEPTH_FOLDER,
            *trial_info,
            f'cam_{CAMERA}',
            'depth',
            '*.png',
        )))

        mp_points = utils_mediapipe.load_points(file_path)
        mp_points = utils_mediapipe.mediapipe_to_unified(
            mp_points.reshape(-1, utils_mediapipe.TOTAL_POINTS_COUNT, 3)
        ).reshape(-1, 3 * utils_unified_format.TOTAL_POINTS_COUNT)
        predicted = None

        if len(mp_points) != len(depth_paths):
            logger.error(
                f'Lenght of points ({len(mp_points)}) and '
                f'depth images ({len(depth_paths)}) does not match!'
            )
            continue

        for points, depth_path in zip(mp_points, depth_paths):
            depth_image = iio.imread(depth_path).T

            if depth_image.max() == 0:
                logger.error(f'Corrupted depth image: {depth_path}')

            frame_points = points.reshape(-1, 3)
            frame_points = camera_systems.zero_points_outside_screen(
                frame_points,
                is_normalized=True,
                inplace=True,
            )
            frame_points = camera_systems.normalized_to_screen(
                frame_points,
                inplace=True,
            )

            depths = depth_extractor.get_depth_in_window(
                depth_image,
                frame_points,
                predicted,
            )

            if predicted is None:
                kalman_filters.reset([
                    np.array([[point], [0]])
                    for point in depths
                ])
            depths_filtered = kalman_filters.update(
                depths,
                use_heuristic=True,
                projection=0,
            )

            # val_acc.append(kalman_filters.filters[7].x[0, 0])
            # val_sigma_acc.append(kalman_filters.filters[7].P[0, 0])

            if NEED_FILTERING:
                predicted = kalman_filters.predict(projection=0)
            depths_filtered = tp.cast(tp.List[float], depths_filtered)
            predicted = tp.cast(tp.List[float], predicted)

            frame_points[:, 2] = depths_filtered
            frame_points = camera_systems.screen_to_world(
                frame_points,
                inplace=True,
            )

        np.save(save_path, mp_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')

        # print('\n Values: \n    ', end='')
        # print(',\n    '.join(str(i) for i in val_acc), end=',\n')
        # print('\n Sigma: \n    ', end='')
        # print(',\n    '.join(str(i) for i in val_sigma_acc), end=',\n')


if __name__ == '__main__':
    main()
