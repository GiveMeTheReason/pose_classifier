import dataclasses
import json
import typing as tp

import numpy as np


ImageSizeT = tp.Tuple[int, int]
IntrinsicT = tp.Tuple[float, float, float, float]


class CameraSystems:
    '''Coordinate systems:
    Normalized(0...1) <-> Screen(0...image_size) <-> World

    Points format: x, y, [depth], ... (extras)
    Points preserves their shape, extras dims are untouched
    '''
    def __init__(
        self,
        image_size: ImageSizeT,
        camera_intrinsic: IntrinsicT
    ) -> None:
        self.image_size = image_size
        self.camera_intrinsic = camera_intrinsic

    def is_points_in_screen(
        self,
        points: np.ndarray,
        is_normalized: bool = False,
    ) -> np.ndarray:
        if not is_normalized:
            points = self.screen_to_normalized(points)
        return np.prod(
            (0 <= points[:, :2]) * (points[:, :2] < 1),
            axis=1,
            dtype=bool,
        )

    def normalized_to_screen(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            points = np.copy(points)
        points[:, :2] *= self.image_size
        return points

    def screen_to_normalized(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            points = np.copy(points)
        points[:, :2] /= self.image_size
        return points

    def screen_to_world(
        self,
        points: np.ndarray,
        depths: tp.Optional[np.ndarray] = None,
        inplace: bool = False,
    ) -> np.ndarray:
        '''If depths is provided, return gets new column with depths
        '''
        if not inplace:
            points = np.copy(points)
        points_depths = depths
        if points_depths is None:
            if points.shape[1] < 3:
                raise Exception('Depth is not provided')
            points_depths = points[:, 2]
        points_depths = points_depths.reshape(-1, 1)
        points[:, :2] -= (self.principal_x, self.principal_y)
        points[:, :2] *= points_depths / (self.focal_x, self.focal_y)
        if depths is None:
            return points
        return np.insert(points, 2, depths, axis=1)

    def world_to_screen(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            points = np.copy(points)
        points_depths = points[:, 2].reshape(-1, 1)
        points[:, :2] *= (self.focal_x, self.focal_y) / points_depths
        points[:, :2] += (self.principal_x, self.principal_y)
        return points

    def normalized_to_world(
        self,
        points: np.ndarray,
        depths: tp.Optional[np.ndarray] = None,
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            points = np.copy(points)
        points = self.normalized_to_screen(points, inplace=True)
        points = self.screen_to_world(points, depths, inplace=True)
        return points

    def world_to_normalized(
        self,
        points: np.ndarray,
        inplace: bool = False,
    ) -> np.ndarray:
        if not inplace:
            points = np.copy(points)
        points = self.world_to_screen(points, inplace=True)
        points = self.screen_to_normalized(points, inplace=True)
        return points

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def focal_x(self) -> float:
        return self.camera_intrinsic[0]

    @property
    def focal_y(self) -> float:
        return self.camera_intrinsic[1]

    @property
    def principal_x(self) -> float:
        return self.camera_intrinsic[2]

    @property
    def principal_y(self) -> float:
        return self.camera_intrinsic[3]


class DepthExtractor:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def get_depth(
        self,
        depth_image: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        points_depths = depth_image[
            points[:, 0].astype(int),
            points[:, 1].astype(int),
        ]
        return points_depths

    def get_depth_in_window(
        self,
        depth_image: np.ndarray,
        points: np.ndarray,
        predicted: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if predicted is None:
            return self.get_depth(depth_image, points)

        low = self.window_size // 2
        high = self.window_size - low

        points_depths = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            x, y = points[i, :2].astype(np.int64)
            x_slice = slice(max(0, x - low), x + high)
            y_slice = slice(max(0, y - low), y + high)
            depth_window = depth_image[x_slice, y_slice]
            idx = [np.abs(depth_window - predicted[i]).argmin()]
            points_depths[i] = depth_window.reshape(-1)[idx]
        return points_depths


def get_camera_params(params_path: str) -> tp.Tuple[ImageSizeT, IntrinsicT]:
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

    image_size: ImageSizeT = (width, height)
    camera_intrinsic: IntrinsicT = (focal_x, focal_y, principal_x, principal_y)

    return image_size, camera_intrinsic


def params_to_intrinsic_matrix(params: IntrinsicT) -> np.ndarray:
    return np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1],
    ])
