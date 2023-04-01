import json
import typing as tp

import numpy as np
import open3d as o3d
import plotly.graph_objs as go


image_sizeT = tp.Tuple[int, int]


def get_camera_params(params_path: str) -> tp.Tuple[image_sizeT, tp.List[float]]:
    with open(params_path) as params_file:
        params = json.load(params_file)
    camera = params['color_camera']
    intrinsic = camera['intrinsics']['parameters']['parameters_as_dict']

    width: int = camera['resolution_width']
    height: int = camera['resolution_height']
    focal_x: float = intrinsic['fx']
    focal_y: float = intrinsic['fy']
    principal_x: float = intrinsic['cx']
    principal_y: float = intrinsic['cy']

    image_size: image_sizeT = (width, height)
    camera_intrinsic = [focal_x, focal_y, principal_x, principal_y]

    return image_size, camera_intrinsic


def camera_params_to_ndarray(params: tp.List[float]) -> np.ndarray:
    return np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1],
    ])


def points_in_screen(points: np.ndarray) -> np.ndarray:
    return np.prod((0 <= points[:, :2]) * (points[:, :2] <= 1), axis=1).astype(bool)


def screen_to_pixel(
    points: np.ndarray,
    width: int,
    height: int,
    with_copy: bool = True
) -> np.ndarray:
    if with_copy:
        points = np.copy(points)
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def attach_depth(
    points: np.ndarray,
    depth_image: np.ndarray,
    with_copy: bool = True
) -> np.ndarray:
    if with_copy:
        points = np.copy(points)
    points[:, 2] = depth_image[points[:, 1].astype(int), points[:, 0].astype(int)]
    return points


def get_mediapipe_points(mp_points_path: str) -> np.ndarray:
    return np.genfromtxt(
        mp_points_path,
        delimiter=',',
        skip_header=1,
        usecols=[*range(99)],
    )


def get_rgbd_image(
    rgb_image_path: str,
    depth_image_path: str,
    depth_scale=1000,
    depth_trunc=5.0,
) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.io.read_image(rgb_image_path),
        depth=o3d.io.read_image(depth_image_path),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def create_point_cloud(
    rgbd_image: o3d.geometry.RGBDImage,
    image_size: image_sizeT,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> o3d.geometry.PointCloud:
    width: int = image_size[0]
    height: int = image_size[1]
    focal_x: float = intrinsic[0, 0]
    focal_y: float = intrinsic[1, 1]
    principal_x: float = intrinsic[0, 2]
    principal_y: float = intrinsic[1, 2]
    params = [width, height, focal_x, focal_y, principal_x, principal_y]

    return o3d.geometry.PointCloud.create_from_rgbd_image(
        image=rgbd_image,
        intrinsic=o3d.camera.PinholeCameraIntrinsic(*params),
        extrinsic=np.linalg.inv(extrinsic),
        project_valid_depth_only=True,
    )


def filter_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    z_min: float = 0.1,
    out_near: int = 20,
    out_radius: float = 0.01,
) -> o3d.geometry.PointCloud:
    shift = np.array([0, 0, z_min])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(point_cloud.points)
    shifted_bounding_box = bounding_box.translate(shift)
    cropped_point_cloud = point_cloud.crop(shifted_bounding_box)

    filtered_point_cloud, _ = cropped_point_cloud.remove_radius_outlier(out_near, out_radius)

    return filtered_point_cloud


def get_scatter_3d(
    points: np.ndarray,
    colors: tp.Optional[np.ndarray] = None,
    size: int = 1,
    step: int = 1,
) -> go.Scatter3d:
    marker: tp.Dict[str, tp.Any] = {'size': size}
    if colors is not None:
        marker['color'] = colors[::step]

    scatter = {
        'x': points[::step, 0],
        'y': points[::step, 1],
        'z': points[::step, 2],
        'mode': 'markers',
        'marker': marker,
    }
    return go.Scatter3d(**scatter)


def visualize_data(
    data: tp.List[tp.Any],
    with_axis: bool = True,
):
    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis),
            )
        )
    )
    fig.update_scenes(aspectmode='data')
    fig.show()
