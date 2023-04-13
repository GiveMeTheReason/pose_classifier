import glob
import os
import typing as tp

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch

import model.classifiers as classifiers
import model.transforms as transforms
import visualizer.utils as utils
from config import CONFIG, TRAIN_CONFIG, VISUALIZER_CONFIG


# [101, 120]
SUBJECT = 101
# ['select', 'call', 'start', 'yes', 'no']
GESTURE = 'select'
# ['both', 'left', 'right']
HAND = 'left'
# [1, 4...6]
TRIAL = 1
# ['center', 'left', 'right']
CAMERA = 'center'
# [0, 120]
FRAME_RANGE = (0, 120)
# True - build PC, false - only mp skeleton
WITH_POINT_CLOUD = False
# True - use raw MP graph, False - use World
USE_MP_RAW = False
# True - from MP, False - from World
TRANSFORM_MP_TO_WORLD = False
# True - mp data with labels, False - without
WITH_LABELS = True
# True - use model, False - don't use
WITH_MODEL = True

if WITH_MODEL:
    exp_id = 3

    device = 'cpu'
    length = 115

    checkpoint_path = os.path.join(
        TRAIN_CONFIG.train_params.output_data,
        f'experiment_{str(exp_id).zfill(3)}',
        'checkpoint.pth',
    )

    with_rejection = TRAIN_CONFIG.gesture_set.with_rejection
    label_map = {gesture: i for i, gesture in enumerate(TRAIN_CONFIG.gesture_set.gestures, start=1)}
    if with_rejection:
        # label_map['_rejection'] = len(label_map)
        label_map['_rejection'] = 0

    inv_label_map = {value: key for key, value in label_map.items()}

    model = classifiers.LSTMClassifier(len(label_map))
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_transforms = transforms.TestTransforms(device=device)


samples_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled

with_rejection = TRAIN_CONFIG.gesture_set.with_rejection
label_map = {gesture: i for i, gesture in enumerate(TRAIN_CONFIG.gesture_set.gestures, start=1)}
if with_rejection:
    # label_map['_rejection'] = len(label_map)
    label_map['_rejection'] = 0

inv_label_map = {value: key for key, value in label_map.items()}

if USE_MP_RAW:
    mp_source_folder = CONFIG.mediapipe.points_pose_raw
else:
    if WITH_LABELS:
        mp_source_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled
    else:
        mp_source_folder = CONFIG.mediapipe.points_pose_world_windowed_filtered

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


class ModelState:
    def __init__(self) -> None:
        self.h_n = None
        self.c_n = None


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


def get_mp_graph(points: np.ndarray) -> np.ndarray:
    graph = np.array([
        8, 6, 5, 4, 0, 1, 2, 3, 7, None,
        10, 9, None,
        12, 11, 23, 24, 12, None,
        12, 14, 16, None,
        11, 13, 15, None,
        22, 16, 18, 20, 16, None,
        21, 15, 17, 19, 15, None,
        24, 26, 28, 30, 32, 28, None,
        23, 25, 27, 29, 31, 27, None,
    ])

    graphed_points = np.zeros((len(graph), 3))
    for i, node in enumerate(graph):
        if node is None:
            graphed_points[i] = np.nan
        else:
            graphed_points[i] = points[node]
    return graphed_points


def get_frame(
    color_path: str,
    depth_path: str,
    frame_points: np.ndarray,
    image_size: utils.image_sizeT,
    intrinsic: np.ndarray,
    frame: int,
    use_mp_online: bool,
    model_state: ModelState,
    mp_solver: tp.Optional[mp.solutions.pose.Pose] = None,
):
    if WITH_POINT_CLOUD:
        depth_image_raw = get_raw_image(depth_path)
        # depth_image_cv2 = get_cv_image(depth_image_path)
        rgb_image_cv2 = get_cv_image(color_path)

        point_cloud = get_point_cloud(color_path, depth_path, image_size, intrinsic)

    if use_mp_online:
        if mp_solver is None:
            raise Exception('Extractor should be provided!')
        frame_points = get_points_from_image(mp_solver, rgb_image_cv2)
    if WITH_LABELS:
        label = frame_points[-1]
        frame_points = frame_points[:-1]
    if WITH_MODEL:
        if WITH_LABELS:
            model_label = torch.tensor([label]) * label_map[GESTURE]
        model_points = np.copy(frame_points[None, ...])
        prediction, model_state.h_n, model_state.c_n = model(test_transforms(model_points), model_state.h_n, model_state.c_n)
        prediction_probs, prediction_label = prediction.max(dim=-1)

    frame_points = frame_points.reshape(-1, 3)
    points_colors = color_points(frame_points, [1, 0, 0])

    # points_pixel = get_pixel_points(frame_points, image_size)
    if TRANSFORM_MP_TO_WORLD:
        frame_points = get_world_points(frame_points, depth_image_raw, intrinsic)
    if not USE_MP_RAW:
        frame_points /= 1000
    else:
        frame_points[:, 2] += 1

    mp_scatter = utils.get_scatter_3d(
        frame_points,
        points_colors,
        size=3,
    )

    mp_graph = utils.get_scatter_3d(
        get_mp_graph(frame_points),
        points_colors,
        mode='lines',
        line=dict(color='darkblue', width=2),
    )

    data = [mp_scatter, mp_graph]
    layout= {}

    if WITH_POINT_CLOUD:
        camera_scatter = utils.get_scatter_3d(
            np.asarray(point_cloud.points),
            np.asarray(point_cloud.colors),
            step=25,
        )
        data.append(camera_scatter)

    if WITH_LABELS:
        if label == 1:
            label_color = np.array([[0, 1, 0]])
        else:
            label_color = np.array([[1, 0, 0]])
        label_scatter = utils.get_scatter_3d(
            np.array([[0, 0, 1]]),
            label_color,
            size=10,
        )
        data.append(label_scatter)

    if WITH_MODEL:
        x_ticks = np.linspace(-0.7, 0.7, len(inv_label_map))
        true_label = inv_label_map[int(model_label.item())]
        model_label = inv_label_map[int(prediction_label.item())]
        scene = {'annotations': [
            {
                'x': x_ticks[i],
                'y': 0.9,
                'z': 1,
                'showarrow': False,
                'text': inv_label_map[i],
            } for i in range(len(inv_label_map))
        ]}
        scene['annotations'].extend([
            {
                'x': -0.5,
                'y': 0,
                'z': 1,
                'showarrow': False,
                'text': f'True label: {true_label}',
            },
            {
                'x': 0.5,
                'y': 0,
                'z': 1,
                'showarrow': False,
                'text': f'Model label: {model_label}',
            }
        ])
        true_label_color = np.array([[0, 0, 1]])
        if true_label == model_label:
            model_label_color = np.array([[0, 1, 0]])
        else:
            model_label_color = np.array([[1, 0, 0]])
        true_label_scatter = utils.get_scatter_3d(
            np.array([[x_ticks[label_map[true_label]], 0.9, 1]]),
            true_label_color,
            size=10,
        )
        model_label_scatter = utils.get_scatter_3d(
            np.array([[x_ticks[label_map[model_label]], 0.9, 1]]),
            model_label_color,
            size=10,
        )
        layout['scene'] = scene
        data.append(true_label_scatter)
        data.append(model_label_scatter)

    go_frame = utils.get_frame(data=data, frame_num=frame, layout=layout)
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
    mp_points_path = os.path.join(
        mp_source_folder,
        f'G{SUBJECT}_{GESTURE}_{HAND}_trial{TRIAL}.npy',
    )
    mp_points = get_points_from_file(mp_points_path)

    image_size, intrinsic = utils.get_camera_params(CONFIG.cameras[f'{CAMERA}_camera_params'])

    model_state = ModelState()

    if WITH_POINT_CLOUD:
        color_paths = sorted(glob.glob(os.path.join(folder_path, 'color', '*.jpg')))
        depth_paths = sorted(glob.glob(os.path.join(folder_path, 'depth', '*.png')))
    else:
        color_paths = [''] * len(mp_points)
        depth_paths = [''] * len(mp_points)

    frame_range = (
        max(FRAME_RANGE[0], 0),
        min(FRAME_RANGE[1], len(mp_points)),
    )

    frames = [get_frame(
            color_paths[frame],
            depth_paths[frame],
            mp_points[frame],
            image_size,
            intrinsic,
            frame,
            use_mp_online,
            model_state,
            mp_solver,
        ) for frame in range(*frame_range)]

    fig = utils.get_figure_3d(len(frames[0].data))
    fig.update(frames=frames)
    fig.update_layout(
        title=mp_points_path,
        scene=VISUALIZER_CONFIG.scene,
        scene_camera=VISUALIZER_CONFIG.scene_camera,
        updatemenus=[VISUALIZER_CONFIG.update_buttons],
        uirevision=True,
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
