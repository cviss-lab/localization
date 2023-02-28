import cv2
import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image
import csv
from scipy.spatial.transform import Rotation

from libs.utils.loader import *
from libs.utils.projection import *


class PreProcessor:
    def __init__(self, input_dir, output_dir, frame_rate=5, depth_max=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.depth_max = depth_max
        self.loader = LocalLoader(".")

    def from_mobile_inspector(self, voxel=0.015, filter_level=2):
        """
        Preprocesses the Mobile Inspector dataset
        :param dataset_dir: path to the Mobile Inspector dataset
        :param output_dir: path to the output directory
        :param depth_max: maximum depth for the depth images
        :param voxel: voxel size for the downsampling of the point cloud
        """

        dataset_dir = self.input_dir
        output_dir = self.output_dir
        depth_max = self.depth_max

        os.makedirs(os.path.join(output_dir, "rgb"))
        os.makedirs(os.path.join(output_dir, "depth"))

        depth_dir = os.path.join(dataset_dir, "depth")
        depth_frames = [
            os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))
        ]

        camera_matrix, rgb_w, rgb_h = self.loader.load_intrinsics(
            os.path.join(dataset_dir, "intrinsics.json")
        )
        poses = self.loader.load_poses(os.path.join(dataset_dir, "poses.csv"))

        shutil.copyfile(
            os.path.join(dataset_dir, "intrinsics.json"),
            os.path.join(output_dir, "intrinsics.json"),
        )
        shutil.copyfile(
            os.path.join(dataset_dir, "poses.csv"),
            os.path.join(output_dir, "poses.csv"),
        )
        shutil.copyfile(
            os.path.join(dataset_dir, "annotations.json"),
            os.path.join(output_dir, "annotations.json"),
        )

        depth_1 = cv2.imread(depth_frames[0])
        depth_width = depth_1.shape[1]
        depth_height = depth_1.shape[0]

        rgb_ratio = rgb_h / rgb_w
        depth_h = int(depth_width * rgb_ratio)
        offset_h = int((depth_h - depth_height) / 2)

        intrinsics = get_intrinsics(
            camera_matrix, depth_width, depth_h, rgb_height=rgb_h, rgb_width=rgb_w
        )
        pc = o3d.geometry.PointCloud()

        for i in range(1, len(depth_frames) + 1):

            print(f"Processing frame {i}/{len(depth_frames)}", end="\r")

            depth_path = os.path.join(dataset_dir, "depth", f"{i}.png")
            if not os.path.exists(depth_path):
                continue

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if np.all(depth == depth.mean()):
                continue

            rgb_path = os.path.join(dataset_dir, "rgb", f"{i}.jpg")
            if not os.path.exists(rgb_path):
                continue

            conf_path = os.path.join(dataset_dir, "confidence", f"{i}.png")
            conf = cv2.imread(conf_path, cv2.IMREAD_UNCHANGED)
            depth[conf < filter_level] = 0

            pose = poses[i]
            T_CW = T_inv(pose2matrix(pose))

            rgb = cv2.imread(rgb_path)

            depth2 = np.zeros((depth_h, depth_width), dtype=np.uint16)
            depth2[offset_h : offset_h + depth_height, :] = depth

            cv2.imwrite(os.path.join(output_dir, "rgb/{:06}.jpg".format(i)), rgb)
            cv2.imwrite(os.path.join(output_dir, "depth/{:06}.png".format(i)), depth2)

            depth_m = depth2.astype(np.float32) / 1000.0
            depth = o3d.geometry.Image(depth_m)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (depth_width, depth_h))
            rgb = np.array(rgb)
            rgb = o3d.geometry.Image(rgb)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb,
                depth,
                depth_scale=1.0,
                depth_trunc=depth_max,
                convert_rgb_to_intensity=False,
            )

            pc += o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsics, extrinsic=T_CW
            )

        if len(pc.points) > 2e7:
            voxel = 2 * voxel
            print("Increasing voxel size to stay under RAM limit..")

        pc_down = pc.voxel_down_sample(voxel)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cloud.pcd"), pc_down)


def get_intrinsics(intrinsics, depth_width, depth_height, rgb_height=1440, rgb_width=1920):
    """
    Scales the intrinsics matrix to be of the appropriate scale for the depth maps.
    """
    intrinsics_scaled = resize_camera_matrix(intrinsics, depth_width / rgb_width, depth_height / rgb_height)
    return o3d.camera.PinholeCameraIntrinsic(width=depth_width, height=depth_height, fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], cx=intrinsics_scaled[0, 2], cy=intrinsics_scaled[1, 2])

def resize_camera_matrix(camera_matrix, scale_x, scale_y):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([[fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]])           