import glob
import os
import typing as tp

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from config import CONFIG
from config import VISUALIZER_CONFIG
import visualizer.utils as utils


# [101, 120]
SUBJECT = 101
# ...
GESTURE = 'call'
# ['both', 'left', 'right']
HAND = 'left'
# [1, 4...6]
TRIAL = 1
# ['center', 'left', 'right']
CAMERA = 'center'
# [0, 120]
FRAME_RANGE = (0, 120)
# True - from MP, False - from World
TRANSFORM_MP_TO_WORLD = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def get_raw_image(image_path: str) -> np.ndarray:
    return iio.imread(image_path)


def get_cv_image(image_path: str) -> cv2.Mat:
    return cv2.imread(image_path)


def get_point_cloud(
    color_image_path: str,
    depth_image_path: str,
    image_size: utils.image_sizeT,
    intrinsic: np.ndarray
) -> utils.PointCloudT:
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
    point_cloud = utils.filter_point_cloud(point_cloud, z_min=1.5)
    return point_cloud


def get_points_from_file(mp_points_path: str) -> np.ndarray:
    mp_points = utils.get_mediapipe_points(mp_points_path)
    return mp_points


def get_points_from_image(solver, image: cv2.Mat) -> np.ndarray:
    landmarks = solver.process(image)
    if landmarks.pose_landmarks is not None:
        frame_points = utils.landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
        frame_points = frame_points.reshape(-1)
        return frame_points
    return np.zeros(0)


def color_points(points: np.ndarray, rgb: tp.Sequence) -> np.ndarray:
    colors = np.zeros_like(points)
    colors[:] = rgb
    return colors


def get_pixel_points(points: np.ndarray, image_size: utils.image_sizeT) -> np.ndarray:
    valid = utils.points_in_screen(points)
    points[~valid] = 0
    points_pixel = utils.screen_to_pixel(points, *image_size)
    return points_pixel


def get_world_points(points: np.ndarray, depth_image: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    valid = utils.points_in_screen(points)
    points[~valid] = 0
    points_world = utils.screen_to_world(points, depth_image, intrinsic)
    return points_world


def get_frame(
    color_path: str,
    depth_path: str,
    frame_points: np.ndarray,
    image_size: utils.image_sizeT,
    intrinsic: np.ndarray,
    frame: int,
    use_mp_online: bool,
    mp_solver: tp.Optional[mp.solutions.pose.Pose] = None,
):
    depth_image_raw = get_raw_image(depth_path)
    # depth_image_cv2 = get_cv_image(depth_image_path)
    rgb_image_cv2 = get_cv_image(color_path)

    point_cloud = get_point_cloud(color_path, depth_path, image_size, intrinsic)

    if use_mp_online:
        if mp_solver is None:
            raise Exception('Extractor should be provided!')
        frame_points = get_points_from_image(mp_solver, rgb_image_cv2)
    frame_points = frame_points.reshape(-1, 3)
    points_colors = color_points(frame_points, [1, 0, 0])

    # points_pixel = get_pixel_points(frame_points, image_size)
    if TRANSFORM_MP_TO_WORLD:
        frame_points = get_world_points(frame_points, depth_image_raw, intrinsic)
    frame_points /= 1000

    mp_scatter = utils.get_scatter_3d(
        frame_points,
        points_colors,
        size=3,
    )

    camera_scatter = utils.get_scatter_3d(
        np.asarray(point_cloud.points),
        np.asarray(point_cloud.colors),
        step=25,
    )

    go_frame = utils.get_frame(data=[mp_scatter, camera_scatter], frame_num=frame)
    return go_frame


def main():
    use_mp_online = False
    if use_mp_online:
        mp_solver = mp_pose.Pose()
    else:
        mp_solver = None

    folder_path = os.path.join(
        CONFIG.dataset.undistorted,
        f'G{str(SUBJECT).zfill(3)}',
        GESTURE,
        HAND,
        f'trial{TRIAL}',
        f'cam_{CAMERA}',
    )
    if TRANSFORM_MP_TO_WORLD:
        mp_source = 'raw'
    else:
        mp_source = 'world'
    mp_points_path = os.path.join(
        CONFIG.mediapipe[f'points_pose_{mp_source}'],
        f'G{SUBJECT}_{GESTURE}_{HAND}_trial{TRIAL}.npy',
    )
    mp_points = get_points_from_file(mp_points_path)

    image_size, intrinsic = utils.get_camera_params(CONFIG.cameras[f'{CAMERA}_camera_params'])

    color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))

    frame_range = (
        max(FRAME_RANGE[0], 0),
        min(FRAME_RANGE[1], len(mp_points)),
    )

    fig = utils.get_figure_3d()
    fig.update_layout(scene=VISUALIZER_CONFIG.scene)
    frames = [get_frame(
            color_paths[frame],
            depth_paths[frame],
            mp_points[frame],
            image_size,
            intrinsic,
            frame,
            use_mp_online,
            mp_solver,
        ) for frame in range(*frame_range)]
    fig.update(frames=frames)
    fig.update_layout(
        scene_camera=VISUALIZER_CONFIG.scene_camera,
        updatemenus=[VISUALIZER_CONFIG.update_buttons],
    )
    fig.show()

    # points_pixel = get_pixel_points(frame_points, image_size)
    # points_world = get_world_points(frame_points, depth_image_raw, intrinsic)

    # plt.scatter(points_pixel[:, 2], points_world[:, 2])
    # plt.xlabel('mediapipe')
    # plt.ylabel('depth')
    # plt.show()

    # for point in points_pixel:
    #     cv2.circle(depth_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
    #     cv2.circle(rgb_image_cv2, (int(point[0]), int(point[1])), 1, color=(0, 0, 255), thickness=-1)
    # cv2.imshow('Depth', depth_image_cv2)
    # cv2.imshow('Color', rgb_image_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
