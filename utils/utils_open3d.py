import typing as tp

import numpy as np
import open3d as o3d


ImageSizeT = tp.Tuple[int, int]
RGBDImageT = o3d.geometry.RGBDImage
PointCloudT = o3d.geometry.PointCloud


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


def get_point_cloud(
    color_image_path: str,
    depth_image_path: str,
    image_size: ImageSizeT,
    intrinsic: np.ndarray
) -> PointCloudT:
    rgbd_image = get_rgbd_image(
        color_image_path,
        depth_image_path,
        depth_trunc=2,
    )
    point_cloud = create_point_cloud(
        rgbd_image,
        image_size,
        intrinsic=intrinsic,
        extrinsic=np.eye(4),
    )
    return point_cloud


def create_point_cloud(
    rgbd_image: RGBDImageT,
    image_size: ImageSizeT,
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
    if len(point_cloud.points) == 0:
        return point_cloud

    shift = np.array([0, 0, z_min])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(point_cloud.points)
    bounding_box.translate(shift)
    cropped_point_cloud = point_cloud.crop(bounding_box)

    filtered_point_cloud, _ = cropped_point_cloud.remove_radius_outlier(out_near, out_radius)

    return filtered_point_cloud
