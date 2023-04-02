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


def get_raw_image(image_path) -> np.ndarray:
    return iio.imread(image_path)


def get_cv_image(image_path) -> cv2.Mat:
    return cv2.imread(image_path)


def get_point_cloud(color_image_path, depth_image_path) -> utils.PointCloudT:
    image_size, intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)

    rgbd_image = utils.get_rgbd_image(
        color_image_path,
        depth_image_path,
        depth_trunc=2,
    )
    point_cloud = utils.create_point_cloud(
        rgbd_image,
        image_size,
        intrinsic=intrinsic,
        extrinsic=np.eye(4),
    )
    point_cloud = utils.filter_point_cloud(point_cloud, z_min=0.5)
    return point_cloud


def get_points_from_file(mp_points_path) -> np.ndarray:
    mp_points = utils.get_mediapipe_points(mp_points_path)
    return mp_points


def get_points_from_image(solver, image) -> np.ndarray:
    landmarks = solver.process(image)
    frame_points = utils.landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
    return frame_points


def color_points(points, rgb) -> np.ndarray:
    colors = np.zeros_like(points)
    colors[:] = rgb
    return colors


def get_pixel_points(points) -> np.ndarray:
    image_size, intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)

    valid = utils.points_in_screen(points)
    points[~valid] = 0
    points_pixel = utils.screen_to_pixel(points, *image_size, False)
    return points_pixel


def get_world_points(points, depth_image) -> np.ndarray:
    image_size, intrinsic = utils.get_camera_params(CONFIG.path.center_camera_params)

    valid = utils.points_in_screen(points)
    points[~valid] = 0
    points_world = utils.screen_to_world(points, depth_image, intrinsic, False)
    return points_world


def main():
    use_mp_online = False
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
        f'G{str(SUBJECT).zfill(3)}',
        GESTURE,
        HAND,
        f'trial{TRIAL}',
        f'cam_{CAMERA}',
    )
    mp_points_path = os.path.join(
        CONFIG.path.mediapipe_points,
        f'G{SUBJECT}_{GESTURE}_{HAND}_trial{TRIAL}.csv',
    )

    color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))

    color_image_path = color_paths[FRAME]
    depth_image_path = depth_paths[FRAME]

    depth_image_raw = get_raw_image(depth_image_path)
    depth_image_cv2 = get_cv_image(depth_image_path)
    rgb_image_cv2 = get_cv_image(color_image_path)

    point_cloud = get_point_cloud(color_image_path, depth_image_path)

    if use_mp_online:
        mp_points = get_points_from_file(mp_points_path)
        frame_points = mp_points[FRAME].reshape(-1, 3)
    else:
        frame_points = get_points_from_image(pose_solver, rgb_image_cv2)
    points_colors = color_points(frame_points, [1, 0, 0])

    points_pixel = get_pixel_points(frame_points)
    points_world = get_world_points(frame_points, depth_image_raw)

    # plt.scatter(points_pixel[:, 2], points_world[:, 2])
    # plt.xlabel('mediapipe')
    # plt.ylabel('depth')
    # plt.show()

    for point in points_pixel:
        cv2.circle(depth_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
        cv2.circle(rgb_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
    cv2.imshow('Depth', depth_image_cv2)
    cv2.imshow('Color', rgb_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mp_scatter = utils.get_scatter_3d(
        points_world,
        points_colors,
        size=1,
    )

    camera_scatter = utils.get_scatter_3d(
        np.asarray(point_cloud.points),
        np.asarray(point_cloud.colors),
        step=1,
    )

    utils.visualize_data([
        mp_scatter,
        camera_scatter,
    ])


if __name__ == '__main__':
    main()
