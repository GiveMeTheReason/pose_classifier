# pyright: reportUnboundVariable=false

import glob
import os
import typing as tp

import cv2
import imageio.v3 as iio
import mediapipe as mp
import numpy as np
import torch

import model.classifiers as classifiers
import model.transforms as transforms
import utils.utils_camera_systems as utils_camera_systems
import utils.utils_kalman_filter as utils_kalman_filter
import utils.utils_mediapipe as utils_mediapipe
import utils.utils_open3d as utils_open3d
import utils.utils_plotly as utils_plotly
from config import DATA_CONFIG, TRAIN_CONFIG, VISUALIZER_CONFIG


SUBJECT = 101
GESTURE = 'select'
HAND = 'left'
TRIAL = 1
CAMERA = 'center'
FRAME_RANGE = (0, 120)
WITH_POINT_CLOUD = False
USE_MP_RAW = False
TRANSFORM_MP_TO_WORLD = False
WITH_LABELS = True
WITH_MODEL = True

label_map = TRAIN_CONFIG.gesture_set.label_map
inv_label_map = TRAIN_CONFIG.gesture_set.inv_label_map

if WITH_MODEL:
    exp_id = 1

    device = 'cpu'
    to_keep = TRAIN_CONFIG.transforms_params.to_keep
    shape_limit = TRAIN_CONFIG.transforms_params.shape_limit

    checkpoint_path = os.path.join(
        TRAIN_CONFIG.train_params.output_data,
        f'experiment_{str(exp_id).zfill(3)}',
        'checkpoint.pth',
    )

    test_transforms = transforms.TestTransforms(
        to_keep=to_keep,
        shape_limit=shape_limit,
        device=device,
    )
    labels_transforms = transforms.LabelsTransforms(
        shape_limit=shape_limit,
        device=device,
    )

    model = classifiers.LSTMClassifier(sum(to_keep), len(label_map))
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

if USE_MP_RAW:
    mp_source_folder = DATA_CONFIG.mediapipe.points_pose_raw
else:
    if WITH_LABELS:
        mp_source_folder = DATA_CONFIG.mediapipe.points_unified_world_filtered_labeled
    else:
        mp_source_folder = DATA_CONFIG.mediapipe.points_unified_world_filtered
# mp_source_folder = DATA_CONFIG.mediapipe.points_pose_raw

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def get_points_from_image(solver, image: cv2.Mat) -> np.ndarray:
    landmarks = solver.process(image)
    if landmarks.pose_landmarks is not None:
        frame_points = utils_mediapipe.landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
        frame_points = frame_points.reshape(-1)
        return frame_points
    return np.zeros(0)


def get_mp_graph(points: np.ndarray) -> np.ndarray:
    graph = np.array([
        8, 6, 5, 4, 0, 1, 2, 3, 7, np.nan,
        10, 9, np.nan,
        12, 11, 23, 24, 12, np.nan,
        12, 14, 16, np.nan,
        11, 13, 15, np.nan,
        22, 16, 18, 20, 16, np.nan,
        21, 15, 17, 19, 15, np.nan,
        24, 26, 28, 30, 32, 28, np.nan,
        23, 25, 27, 29, 31, 27, np.nan,
    ])

    graphed_points = np.zeros((len(graph), 3))
    for i, node in enumerate(graph):
        if np.isnan(node):
            graphed_points[i] = np.nan
        else:
            graphed_points[i] = points[int(node)]
    return graphed_points


def get_frame(
    color_path: str,
    depth_path: str,
    frame_points: np.ndarray,
    cam_sys: utils_camera_systems.CameraSystems,
    frame: int,
    use_mp_online: bool,
    mp_solver: tp.Optional[mp.solutions.pose.Pose] = None,
):
    if WITH_POINT_CLOUD:
        depth_image_raw = iio.imread(depth_path)
        # depth_image_cv2 = cv2.imread(depth_image_path)
        rgb_image_cv2 = cv2.imread(color_path)

        point_cloud = utils_open3d.get_point_cloud(
            color_path,
            depth_path,
            cam_sys.image_size,
            utils_camera_systems.params_to_intrinsic_matrix(cam_sys.camera_intrinsic),
        )
        point_cloud = utils_open3d.filter_point_cloud(point_cloud, z_min=1.5)

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
        prediction = model(test_transforms(model_points), use_hidden=True)
        prediction_probs, prediction_label = prediction.max(dim=-1)

    frame_points = frame_points.reshape(-1, 3)
    points_colors = np.zeros_like(frame_points)
    points_colors[:] = [1, 0, 0]

    if TRANSFORM_MP_TO_WORLD:
        frame_points = cam_sys.zero_points_outside_screen(frame_points, is_normalized=True)
        frame_points = cam_sys.normalized_to_world(frame_points, depth_image_raw)
    if not USE_MP_RAW:
        frame_points /= 1000
    else:
        frame_points[:, 2] += 1

    mp_scatter = utils_plotly.get_scatter_3d(
        frame_points,
        points_colors,
        size=3,
    )

    mp_graph = utils_plotly.get_scatter_3d(
        get_mp_graph(frame_points),
        points_colors,
        mode='lines',
        line=dict(color='darkblue', width=2),
    )

    data = [mp_scatter, mp_graph]
    layout= {}

    if WITH_POINT_CLOUD:
        camera_scatter = utils_plotly.get_scatter_3d(
            np.asarray(point_cloud.points),
            np.asarray(point_cloud.colors),
            step=10,
        )
        data.append(camera_scatter)

    if WITH_LABELS and not WITH_MODEL:
        if label == 1:
            label_color = np.array([[0, 1, 0]])
        else:
            label_color = np.array([[1, 0, 0]])
        label_scatter = utils_plotly.get_scatter_3d(
            np.array([[0, 0, 1]]),
            label_color,
            size=5,
        )
        data.append(label_scatter)

    if WITH_MODEL:
        x_ticks = np.linspace(-0.7, 0.7, len(inv_label_map))
        if WITH_LABELS:
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
                'x': 0.5,
                'y': 0.9,
                'z': 2,
                'showarrow': False,
                'text': f'Model label: {model_label}',
            },
        ])
        model_label_color = np.array([[0, 1, 0]])
        if WITH_LABELS:
            scene['annotations'].extend([
                {
                    'x': 0.5,
                    'y': 0.8,
                    'z': 2,
                    'showarrow': False,
                    'text': f'True label: {true_label}',
                },
            ])
            true_label_color = np.array([[0, 0, 1]])
            if true_label == model_label:
                model_label_color = np.array([[0, 1, 0]])
            else:
                model_label_color = np.array([[1, 0, 0]])
            true_label_scatter = utils_plotly.get_scatter_3d(
                np.array([[x_ticks[label_map[true_label]], 0.9, 1]]),
                true_label_color,
                size=5,
            )
            data.append(true_label_scatter)
        model_label_scatter = utils_plotly.get_scatter_3d(
            np.array([[x_ticks[label_map[model_label]], 0.9, 1]]),
            model_label_color,
            size=5,
        )
        data.append(model_label_scatter)
        layout['scene'] = scene

    go_frame = utils_plotly.get_frame(data=data, frame_num=frame, layout=layout)
    return go_frame


def main():
    use_mp_online = False
    if use_mp_online:
        mp_solver = mp_pose.Pose()
    else:
        mp_solver = None

    folder_path = os.path.join(
        DATA_CONFIG.dataset.undistorted,
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
    mp_points = utils_mediapipe.load_points(mp_points_path)

    image_size, intrinsic = utils_camera_systems.get_camera_params(DATA_CONFIG.cameras[f'{CAMERA}_camera_params'])
    cam_sys = utils_camera_systems.CameraSystems(image_size, intrinsic)

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
            cam_sys,
            frame,
            use_mp_online,
            mp_solver,
        ) for frame in range(*frame_range)]

    fig = utils_plotly.create_figure_3d(len(frames[0].data))
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
