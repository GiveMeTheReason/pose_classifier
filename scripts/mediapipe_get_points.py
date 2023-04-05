import glob
import logging
import os
import sys

import imageio.v3 as iio
import mediapipe as mp
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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


GESTURES = (
    'select',
    'call',
    'start',
    'yes',
    'no',
)
CAMERA = 'center'
FORCE = False


def main():
    logger.info('Starting mp labeling script...')

    mp_solver_settings = dict(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # mp_solver = mp_pose.Pose(**mp_solver_settings)
    # mp_solver = mp_holistic.Holistic(**mp_solver_settings)

    folder_paths = []
    for gesture in GESTURES:
        folder_paths.extend(sorted(glob.glob(os.path.join(
            CONFIG.dataset.undistorted,
            'G*',
            gesture,
            '*',
            'trial*',
            f'cam_{CAMERA}',
        ))))
    path_depth = len(CONFIG.dataset.undistorted.split(os.path.sep))

    logger.info(f'Found {len(folder_paths)} trials')

    with open(CONFIG.mediapipe.columns_pose) as file:
        csv_header = [line.strip() for line in file]
    csv_header_str = ','.join(csv_header)

    for counter, trial_path in enumerate(folder_paths, start=1):
        logger.info(f'Start processing {counter}/{len(folder_paths)} trial: {trial_path}')

        color_paths = sorted(glob.glob(os.path.join(trial_path, 'color', '*.jpg')))
        # depth_paths = sorted(glob.glob(os.path.join(trial_path, 'depth', '*.png')))

        path_info = color_paths[0].split(os.path.sep)
        path_info[path_depth+1] = path_info[path_depth+1].replace('_', '-')
        save_path = os.path.join(
            CONFIG.mediapipe.points_pose_raw,
            '_'.join(path_info[path_depth:path_depth+4]) + '.npy',
        )

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        mp_solver = mp_pose.Pose(**mp_solver_settings)

        trial_points = np.zeros((len(color_paths), 33*3))

        for i, image_path in enumerate(color_paths):
            color_image = iio.imread(image_path)

            landmarks = mp_solver.process(color_image)
            if landmarks.pose_landmarks is not None:
                frame_points = utils.landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
                trial_points[i] = frame_points.reshape(-1)
            else:
                trial_points[i] = 0.0

        np.save(save_path, trial_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
