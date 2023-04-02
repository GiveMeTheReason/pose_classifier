import glob
import os

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from config import CONFIG
import visualizer.utils as utils


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


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

    color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
    # color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.png')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))

    color_image_path = color_paths[FRAME]
    depth_image_path = depth_paths[FRAME]

    # rgbd_image = utils.get_rgbd_image(
    #     color_image_path,
    #     depth_image_path,
    #     depth_trunc=2,
    # )
    depth_image = iio.imread(depth_image_path)
    depth_image_cv2 = cv2.imread(depth_image_path)
    rgb_image_cv2 = cv2.imread(color_image_path)

    image_size, camera_intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)
    intrinsic = utils.camera_params_to_ndarray(camera_intrinsic)

    # point_cloud = utils.create_point_cloud(
    #     rgbd_image,
    #     image_size,
    #     intrinsic=intrinsic,
    #     extrinsic=np.eye(4),
    # )
    # point_cloud = utils.filter_point_cloud(point_cloud, z_min=0.5)

    # mp_points_path = os.path.join(
    #     CONFIG.path.mediapipe_points,
    #     f'02.G{SUBJECT}-parsed',
    #     f'{GESTURE}_{HAND}_trial{TRIAL}',
    #     'joints.csv',
    # )

    # mp_points = utils.get_mediapipe_points(mp_points_path)
    # frame_points = mp_points[FRAME].reshape(-1, 3)

    landmarks = pose_solver.process(rgb_image_cv2)
    frame_points = utils.landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]

    mp_colors = np.zeros_like(frame_points)
    mp_colors[:, 0] = 1

    valid = utils.points_in_screen(frame_points)
    frame_points[~valid] = 0
    utils.screen_to_pixel(frame_points, *image_size, False)
    a_mean = np.mean(frame_points[valid, 2])
    a = [i / a_mean for i in frame_points[:, 2]]
    # utils.screen_to_pixel(frame_points, *depth_image.shape[::-1], False)
    utils.attach_depth(frame_points, depth_image, False)
    b_mean = np.mean(frame_points[valid, 2] / 1000)
    b = [(i / b_mean) / 1000 for i in frame_points[:, 2]]
    plt.scatter(a, b)
    plt.xlabel('mediapipe')
    plt.ylabel('depth')
    plt.show()

    for point in frame_points:
        cv2.circle(depth_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
        cv2.circle(rgb_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Depth', depth_image_cv2)
    cv2.imshow('Color', rgb_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mp_scatter = utils.get_scatter_3d(
        frame_points,
        mp_colors,
        size=1,
    )

    # camera_scatter = utils.get_scatter_3d(
    #     np.asarray(point_cloud.points),
    #     np.asarray(point_cloud.colors),
    #     step=1,
    # )

    utils.visualize_data([
        mp_scatter,
        # camera_scatter,
    ])


if __name__ == '__main__':
    main()
