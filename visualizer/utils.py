import json
import os
import typing as tp

import numpy as np
import open3d as o3d
import plotly.graph_objs as go


image_sizeT = tp.Tuple[int, int]
RGBDImageT = o3d.geometry.RGBDImage
PointCloudT = o3d.geometry.PointCloud


class DepthExtractor:
    def __init__(self, width: int, height: int, intrinsic: np.ndarray) -> None:
        self.width = width
        self.height = height
        self.intrinsic = intrinsic

        self.is_inited: bool = False
        self.prev_depth: np.ndarray = np.array([0])
        self.velocity: np.ndarray = np.array([0])

    def points_in_screen(points: np.ndarray) -> np.ndarray:
        return np.prod((0 <= points[:, :2]) * (points[:, :2] <= 1), axis=1).astype(bool)

    def screen_to_pixel(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ):
        if not inplace:
            points = np.copy(points)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        if not inplace:
            return points

    def attach_depth(
        self,
        points: np.ndarray,
        depth_image: np.ndarray,
        inplace: bool = False,
    ):
        if not inplace:
            points = np.copy(points)
        points[:, 2] = depth_image[points[:, 1].astype(int), points[:, 0].astype(int)]
        if not inplace:
            return points

    def attach_depth_in_window(
        self,
        points: np.ndarray,
        depth_image: np.ndarray,
        inplace: bool = False,
    ):
        if not inplace:
            points = np.copy(points)

        if not self.is_inited:
            self.attach_depth(points, depth_image, True)
            self.prev_depth = np.copy(points[:, 2])
            self.velocity = np.zeros_like(self.prev_depth)
            self.is_inited = True
            if not inplace:
                return points
            return

        for i in range(points.shape[0]):
            x, y = points[i, 0], points[i, 1]
            depth_window = depth_image[max(0, y-2):y+3, max(x-2, 0):x+3]
            points[i, 2] = depth_window.reshape(-1)[[np.abs(depth_window - self.predicted[i]).argmin()]]
        self.velocity = points[:, 2] - self.prev_depth
        self.prev_depth = np.copy(points[:, 2])
        if not inplace:
            return points

    def pixel_to_world(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ):
        if not inplace:
            points = np.copy(points)
        points[:, 0] = (points[:, 0] - self.principal_x) * points[:, 2] / self.focal_x
        points[:, 1] = (points[:, 1] - self.principal_y) * points[:, 2] / self.focal_y
        if not inplace:
            return points

    def screen_to_world(
        self,
        points: np.ndarray,
        depth_image: np.ndarray,
        windowed: bool = False,
        inplace: bool = False,
    ):
        if not inplace:
            points = np.copy(points)
        self.screen_to_pixel(points, True)
        self.attach_depth(points, depth_image, True)
        self.pixel_to_world(points, True)
        if not inplace:
            return points

    @property
    def focal_x(self) -> float:
        return self.intrinsic[0, 0]

    @property
    def focal_y(self) -> float:
        return self.intrinsic[1, 1]

    @property
    def principal_x(self) -> float:
        return self.intrinsic[0, 2]

    @property
    def principal_y(self) -> float:
        return self.intrinsic[1, 2]

    @property
    def predicted(self) -> float:
        return self.prev_depth + self.velocity


def get_camera_params(params_path: str) -> tp.Tuple[image_sizeT, np.ndarray]:
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

    return image_size, camera_params_to_ndarray(camera_intrinsic)


def camera_params_to_ndarray(params: tp.List[float]) -> np.ndarray:
    return np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1],
    ])


def points_in_screen(points: np.ndarray) -> np.ndarray:
    return np.prod((0 <= points[:, :2]) * (points[:, :2] <= 1), axis=1).astype(bool)


@tp.overload
def screen_to_pixel(points: np.ndarray, width: int, height: int, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def screen_to_pixel(points: np.ndarray, width: int, height: int, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def screen_to_pixel(
    points: np.ndarray,
    width: int,
    height: int,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    points[:, 0] *= width
    points[:, 1] *= height
    if not inplace:
        return points


@tp.overload
def attach_depth(points: np.ndarray, depth_image: np.ndarray, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def attach_depth(points: np.ndarray, depth_image: np.ndarray, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def attach_depth(
    points: np.ndarray,
    depth_image: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    points[:, 2] = depth_image[points[:, 1].astype(int), points[:, 0].astype(int)]
    if not inplace:
        return points


@tp.overload
def attach_depth_in_window(points: np.ndarray, depth_image: np.ndarray, predicted: np.ndarray, inplace: tp.Literal[True]) -> np.ndarray: ...
@tp.overload
def attach_depth_in_window(points: np.ndarray, depth_image: np.ndarray, predicted: np.ndarray, inplace: tp.Literal[False] = False) -> tp.Tuple[np.ndarray, np.ndarray]: ...
def attach_depth_in_window(
    points: np.ndarray,
    depth_image: np.ndarray,
    predicted: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)

    for i in range(points.shape[0]):
        x, y = points[i, 0], points[i, 1]
        depth_window = depth_image[max(0, y-2):y+3, max(x-2, 0):x+3]
        points[i, 2] = depth_window.reshape(-1)[[np.abs(depth_window - predicted[i]).argmin()]]

    if not inplace:
        return points, predicted
    return predicted


@tp.overload
def pixel_to_world(points: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def pixel_to_world(points: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def pixel_to_world(
    points: np.ndarray,
    intrinsic: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    focal_x: float = intrinsic[0, 0]
    focal_y: float = intrinsic[1, 1]
    principal_x: float = intrinsic[0, 2]
    principal_y: float = intrinsic[1, 2]
    points[:, 0] = (points[:, 0] - principal_x) * points[:, 2] / focal_x
    points[:, 1] = (points[:, 1] - principal_y) * points[:, 2] / focal_y
    if not inplace:
        return points


@tp.overload
def screen_to_world(points: np.ndarray, depth_image: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def screen_to_world(points: np.ndarray, depth_image: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def screen_to_world(
    points: np.ndarray,
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)

    height, width = depth_image.shape
    screen_to_pixel(points, width, height, True)
    attach_depth(points, depth_image, True)
    pixel_to_world(points, intrinsic, True)

    if not inplace:
        return points


def get_mediapipe_points_csv(mp_points_path: str) -> np.ndarray:
    return np.genfromtxt(
        mp_points_path,
        delimiter=',',
        skip_header=1,
        usecols=[*range(99)],
    )


def get_mediapipe_points_npy(mp_points_path: str) -> np.ndarray:
    return np.load(mp_points_path, allow_pickle=True)


def get_mediapipe_points(mp_points_path: str) -> np.ndarray:
    file_format = os.path.splitext(mp_points_path)[-1]
    if file_format == '.csv':
        return get_mediapipe_points_csv(mp_points_path)
    if file_format == '.npy':
        return get_mediapipe_points_npy(mp_points_path)
    raise Exception('Unknown data format')


def landmarks_to_array(landmarks) -> np.ndarray:
    result = np.zeros((len(landmarks), 4))
    for i, landmark in enumerate(landmarks):
        result[i] = landmark.x, landmark.y, landmark.z, landmark.visibility
    return result


def get_rgbd_image(
    rgb_image_path: str,
    depth_image_path: str,
    depth_scale=1000,
    depth_trunc=5.0,
) -> RGBDImageT:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.io.read_image(rgb_image_path),
        depth=o3d.io.read_image(depth_image_path),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def create_point_cloud(
    rgbd_image: RGBDImageT,
    image_size: image_sizeT,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> PointCloudT:
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
    point_cloud: PointCloudT,
    z_min: float = 0.1,
    out_near: int = 20,
    out_radius: float = 0.01,
) -> PointCloudT:
    shift = np.array([0, 0, z_min])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(point_cloud.points)
    shifted_bounding_box = bounding_box.translate(shift)
    cropped_point_cloud = point_cloud.crop(shifted_bounding_box)

    filtered_point_cloud, _ = cropped_point_cloud.remove_radius_outlier(out_near, out_radius)

    return filtered_point_cloud


def get_figure_3d(
    with_axis: bool = True,
) -> go.Figure:
    empty_scatter = get_scatter_3d(np.zeros((1, 3)))
    fig = go.Figure(
        data=[empty_scatter, empty_scatter],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis),
            )
        )
    )
    return fig


def get_frame(data: tp.Any, frame_num: int) -> go.Frame:
    frame = go.Frame(
        data=data,
        traces=[0, 1],
        name=f'frame{frame_num}'
    )
    return frame


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
