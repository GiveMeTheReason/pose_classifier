import json
import typing as tp

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


image_sizeT = tp.Tuple[int, int]


def get_camera_params(params_path: str) -> tp.Tuple[image_sizeT, tp.List[float]]:
    with open(params_path, 'r') as params_file:
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


def visualize_point_cloud(
    points: np.ndarray,
    colors: tp.Optional[np.ndarray] = None,
    step: int = 1,
    with_axis: bool = True,
):
    marker: tp.Dict[str, tp.Any] = {'size': 1}
    if colors is not None:
        marker['color'] = colors[::step]

    scatter = {
        'x': points[::step, 0],
        'y': points[::step, 1],
        'z': points[::step, 2],
        'mode': 'markers',
        'marker': marker,
    }

    fig = go.Figure(
        data=[
            go.Scatter3d(**scatter),
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis)
            )
        )
    )

    fig.show()
