import glob
import os
import time

import cv2
import imageio.v3 as iio
import mediapipe as mp
import numpy as np

from config import CONFIG
import visualizer.utils as utils


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def landmarks_to_array(landmarks) -> np.ndarray:
    result = np.zeros((len(landmarks), 4))
    for i, landmark in enumerate(landmarks):
        result[i] = landmark.x, landmark.y, landmark.z, landmark.visibility
    return result


def main():
    pose_solver = mp_pose.Pose()

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
        # CONFIG.path.sample_data,
        f'G{str(SUBJECT).zfill(3)}',
        GESTURE,
        HAND,
        f'trial{TRIAL}',
        f'cam_{CAMERA}',
        # CAMERA,
    )

    start_time = time.time()

    color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
    # color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.png')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))

    image_size, camera_intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)
    intrinsic = utils.camera_params_to_ndarray(camera_intrinsic)

    for color_image_path, depth_image_path in zip(color_paths, depth_paths):
        depth_image = iio.imread(depth_image_path)
        depth_image_cv2 = cv2.imread(depth_image_path)
        rgb_image_cv2 = cv2.imread(color_image_path)

        landmarks = pose_solver.process(rgb_image_cv2)
        frame_points = landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]

        mp_colors = np.zeros_like(frame_points)
        mp_colors[:, 0] = 1

        valid = utils.points_in_screen(frame_points)
        frame_points[~valid] = 0
        utils.screen_to_pixel(frame_points, *image_size, False)
        utils.attach_depth(frame_points, depth_image, False)

    end_time = time.time()

    print(f'1 trial time: {end_time - start_time :.2f}')


if __name__ == '__main__':
    main()
