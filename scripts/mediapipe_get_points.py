import itertools
import glob
import os
import tqdm

import imageio.v3 as iio
import mediapipe as mp
import numpy as np

import utils.utils_logging as utils_logging
import utils.utils_mediapipe as utils_mediapipe
from config import DATA_CONFIG, TRAIN_CONFIG


logger = utils_logging.init_logger(__name__)

# mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

DEPTH_FOLDER = DATA_CONFIG.dataset.undistorted
SAVE_FOLDER = DATA_CONFIG.mediapipe.points_holistic_raw
GESTURES = TRAIN_CONFIG.gesture_set.gestures
CAMERA = 'center'
FORCE = False


def main():
    logger.info('Starting MediaPipe labeling script...')

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    mp_solver_settings = dict(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # mp_solver = mp_pose.Pose(**mp_solver_settings)
    mp_solver = mp_holistic.Holistic(**mp_solver_settings)

    path_depth = len(DEPTH_FOLDER.split(os.path.sep))
    folder_paths = []
    for gesture in GESTURES:
        folder_paths.extend(sorted(glob.glob(os.path.join(
            DEPTH_FOLDER,
            'G*',
            gesture,
            '*',
            'trial*',
            f'cam_{CAMERA}',
        ))))

    logger.info(f'Found {len(folder_paths)} trials')

    for trial_path in tqdm.tqdm(folder_paths):
        color_paths = sorted(glob.glob(os.path.join(trial_path, 'color', '*.jpg')))
        # depth_paths = sorted(glob.glob(os.path.join(trial_path, 'depth', '*.png')))

        path_info = color_paths[0].split(os.path.sep)
        path_info[path_depth+1] = path_info[path_depth+1].replace('_', '-')
        save_path = os.path.join(
            SAVE_FOLDER,
            '_'.join(path_info[path_depth:path_depth+4]) + '.npy',
        )

        if not FORCE and os.path.exists(save_path):
            logger.info(f'Already exists, skipped: {save_path}')
            continue

        mp_solver.reset()

        trial_points = np.zeros((len(color_paths), (utils_mediapipe.TOTAL_POINTS_COUNT) * 3))

        for i, image_path in enumerate(color_paths):
            color_image = iio.imread(image_path)

            landmarks = mp_solver.process(color_image)

            joined_landmarks = itertools.chain(
                landmarks.pose_landmarks.landmark if landmarks.pose_landmarks is not None else utils_mediapipe.EMPTY_POSE,
                landmarks.left_hand_landmarks.landmark if landmarks.left_hand_landmarks is not None else utils_mediapipe.EMPTY_HAND,
                landmarks.right_hand_landmarks.landmark if landmarks.right_hand_landmarks is not None else utils_mediapipe.EMPTY_HAND,
            )
            frame_points = utils_mediapipe.landmarks_to_array(joined_landmarks)[:, :3]
            trial_points[i] = frame_points.reshape(-1)

        np.save(save_path, trial_points, fix_imports=False)

        logger.info(f'Saved at: {save_path}')


if __name__ == '__main__':
    main()
