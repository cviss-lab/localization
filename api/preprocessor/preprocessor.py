import cv2
import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image
import csv
from scipy.spatial.transform import Rotation
import json

from libs.utils.loader import *
from libs.utils.strayscanner import *
from libs.utils.projection import *


class PreProcessor:
    def __init__(self, input_dir, output_dir, frame_rate=5, depth_max=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.depth_max = depth_max
        self.loader = LocalLoader(".")

    def complete_depth(self, full=False):
        """
        Completes depth maps for all images in the dataset using cloud.ply
        :param data_folder: path to the dataset folder
        """
        data_folder = self.output_dir

        os.makedirs(os.path.join(data_folder, "depth"), exist_ok=True)
        pointcloud = self.loader.load_pc(os.path.join(data_folder, "cloud.ply"))

        poses = self.loader.load_poses(os.path.join(data_folder, "poses.csv"))
        depth_dict = self.loader.load_imgs_dict(
            poses, os.path.join(data_folder, "depth")
        )
        depth_1 = list(depth_dict.values())[0]
        depth_1 = cv2.imread(
            os.path.join(data_folder, "depth", depth_1), cv2.IMREAD_UNCHANGED
        )
        depth_h, depth_w = depth_1.shape

        K, image_width, image_height = self.loader.load_intrinsics(
            os.path.join(data_folder, "intrinsics.json")
        )
        Ks = resize_camera_matrix(K, depth_w / image_width, depth_h / image_height)

        for i, img in enumerate(depth_dict):
            print(f"Completing depth frame {i}/{len(poses.keys())}", end="\r")
            pose = poses[img]
            D = cv2.imread(
                os.path.join(data_folder, "depth", depth_dict[img]),
                cv2.IMREAD_UNCHANGED,
            )
            D_cloud = cloud_to_depth(
                pointcloud, Ks, pose, w=depth_w, h=depth_h, point_size=2
            )
            if full:
                D = D_cloud
            else:
                D[D == 0] = D_cloud[D == 0]
            cv2.imwrite(
                os.path.join(
                    data_folder, os.path.join(data_folder, "depth", depth_dict[img])
                ),
                D,
            )

    def from_lidar(self, scale=0.5):
        """
        Precalculates depth maps for all images in the dataset and saves them as png images
        :param data_folder: path to the dataset folder
        :param scale: scale factor for the depth map
        """
        data_folder = self.input_dir
        output_dir = self.output_dir

        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

        pointcloud = self.loader.load_pc(os.path.join(data_folder, "cloud.ply"))
        K, image_width, image_height = self.loader.load_intrinsics(
            os.path.join(data_folder, "intrinsics.json")
        )

        poses = self.loader.load_poses(os.path.join(data_folder, "poses.csv"))
        rgb_dict = self.loader.load_imgs_dict(poses, os.path.join(data_folder, "rgb"))

        for i, img in enumerate(rgb_dict):
            print(f"Processing frame {i}", end="\r")
            pose = poses(img)
            D = cloud_to_depth(
                pointcloud,
                K,
                pose,
                w=image_width,
                h=image_height,
                s=scale,
                point_size=3,
            )
            cv2.imwrite(os.path.join(output_dir, "depth/{:06}.png".format(i + 1)), D)

    def from_one3d(self):

        dataset_dir = self.input_dir
        output_dir = self.output_dir        
        

        poses = []

        # shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(os.path.join(output_dir, "rgb"))
        os.makedirs(os.path.join(output_dir, "depth"))       

        
        with open(os.path.join(dataset_dir, 'intrinsics.json')) as f:
            intrinsics = json.load(f)
        camera_matrix = np.array(intrinsics["camera_matrix"])
        w = float(intrinsics["width"])
        h = float(intrinsics["height"])

        pointcloud = self.loader.load_pc(os.path.join(dataset_dir, "cloud.ply"))
        # o3d.visualization.draw_geometries([pointcloud])

        with open(os.path.join(dataset_dir, 'poses.txt')) as f:
            Lines = f.readlines()
            
        for i,l in enumerate(Lines):
            print(f"Processing frame {i}", end="\r")
            line = l.split(' ')
            img = cv2.imread(os.path.join(dataset_dir, 'images', line[0]))

            R = np.array(line[4:], dtype=np.float64).reshape(3,3).T
            q = Rotation.from_matrix(R).as_quat()

            poses.append(
                [
                    i,
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                    q[0],
                    q[1],
                    q[2],
                    q[3],
                ]
            )

            D = cloud_to_depth(
                pointcloud,
                camera_matrix,
                poses[-1][1:],
                w=w,
                h=h,
                s=1,
                point_size=2,
            )
            # print(D)
            cv2.imwrite(os.path.join(output_dir, "rgb/{:06}.png".format(i + 1)), img)                        
            cv2.imwrite(os.path.join(output_dir, "depth/{:06}.png".format(i + 1)), D)                        

        with open(os.path.join(output_dir, "poses.csv"), mode="w") as file:
            writer = csv.writer(file, delimiter=",")
            for row in poses:
                writer.writerow(row)

        with open(os.path.join(output_dir, 'intrinsics.json')) as f:
            json.dump(intrinsics, f)


    def from_stray_scanner(self, resize=1, voxel=0.015, filter_level=2):
        """
        Preprocesses the Stray Scanner dataset
        :param dataset_dir: path to the Stray Scanner dataset
        :param output_dir: path to the output directory
        :param depth_max: maximum depth for the depth images
        :param frame_rate: frame rate of the output images
        :param voxel: voxel size for the downsampling of the point cloud
        :param resize: scale factor for the rgb images
        """

        dataset_dir = self.input_dir
        output_dir = self.output_dir
        depth_max = self.depth_max
        frame_rate = self.frame_rate

        camera_matrix = np.loadtxt(
            os.path.join(dataset_dir, "camera_matrix.csv"), delimiter=","
        )
        odometry = np.loadtxt(
            os.path.join(dataset_dir, "odometry.csv"), delimiter=",", skiprows=1
        )
        transforms = []
        poses = []

        # shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(os.path.join(output_dir, "rgb"))
        os.makedirs(os.path.join(output_dir, "depth"))

        for line in odometry:
            # timestamp, frame, x, y, z, qx, qy, qz, qw
            position = line[2:5]
            quaternion = line[5:]
            T_WC = np.eye(4)
            T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            T_WC[:3, 3] = position
            transforms.append(T_WC)

        depth_dir = os.path.join(dataset_dir, "depth")
        depth_frames = [
            os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))
        ]
        depth_frames = [f for f in depth_frames if ".npy" in f or ".png" in f]

        depth_1 = cv2.imread(depth_frames[0])
        depth_width = depth_1.shape[1]
        depth_height = depth_1.shape[0]

        intrinsics = get_intrinsics(camera_matrix, depth_width, depth_height)
        pc = o3d.geometry.PointCloud()
        rgb_path = os.path.join(dataset_dir, "rgb.mp4")
        video = cv2.VideoCapture(rgb_path)

        keep_every_frames = max([1, int(video.get(cv2.CAP_PROP_FPS) / frame_rate)])
        i_frame = 1
        n_frames = int(len(transforms) / keep_every_frames)
        for i, T_WC in enumerate(transforms):
            ret, rgb = video.read()
            if not ret:
                continue
            if i % keep_every_frames != 0:
                continue
            print(f"Processing frame {i_frame}/{n_frames}", end="\r")
            T_CW = np.linalg.inv(T_WC)
            confidence = load_confidence(
                os.path.join(dataset_dir, "confidence", f"{i:06}.png")
            )
            depth_path = depth_frames[i]
            depth, depth_mm = load_depth(
                depth_path, confidence, filter_level=filter_level
            )

            rgb_h = int(rgb.shape[0] * resize)
            rgb_w = int(rgb.shape[1] * resize)

            rgb_cv = cv2.resize(rgb, (rgb_w, rgb_h))

            cv2.imwrite(
                os.path.join(output_dir, "rgb/{:06}.jpg".format(i_frame)), rgb_cv
            )
            cv2.imwrite(
                os.path.join(output_dir, "depth/{:06}.png".format(i_frame)), depth_mm
            )

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            rgb = Image.fromarray(rgb)
            rgb = rgb.resize((depth_width, depth_height))
            rgb = np.array(rgb)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb),
                depth,
                depth_scale=1.0,
                depth_trunc=depth_max,
                convert_rgb_to_intensity=False,
            )
            pc += o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsics, extrinsic=T_CW
            )

            line = odometry[i]
            position = line[2:5]
            quaternion = line[5:]
            poses.append(
                [
                    i_frame,
                    position[0],
                    position[1],
                    position[2],
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                ]
            )
            i_frame += 1

        with open(os.path.join(output_dir, "poses.csv"), mode="w") as file:
            writer = csv.writer(file, delimiter=",")
            for row in poses:
                writer.writerow(row)

        camera_matrix_cv = resize_camera_matrix(camera_matrix, resize, resize)
        intrinsics = {
            "camera_matrix": camera_matrix_cv.tolist(),
            "dist_coeff": [0, 0, 0, 0, 0],
            "height": rgb_h,
            "width": rgb_w,
        }
        with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
            json.dump(intrinsics, f)

        if len(pc.points) > 2e7:
            voxel = 2 * voxel
            print("Increasing voxel size to stay under RAM limit..")

        pc_down = pc.voxel_down_sample(voxel)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cloud.ply"), pc_down)

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
        o3d.io.write_point_cloud(os.path.join(output_dir, "cloud.ply"), pc_down)

        # pick ground plane points normal to z axis (id, x, y, z)
        picking_list = np.array([[-1, 1, 0, 0], [-1, 2, 0, 0], [-1, 1, 1, 0]]).reshape(
            (-1, 4)
        )
        np.savetxt(
            os.path.join(output_dir, "picking_list.txt"),
            picking_list,
            fmt="%d",
            delimiter=",",
        )

    def cloud_exists(self):
        """
        Checks if the pointcloud exists (cloud.ply) in the output_dir
        :return: Boolean
        """
        return os.path.isfile(os.path.join(self.output_dir, "cloud.ply"))

    def save_plan_view(self, height: float):
        g_plane = calc_gnd_plane(
            self.loader.load_gnd_pts(os.path.join(self.output_dir, "picking_list.txt"))
        )
        map = self.loader.load_pc(os.path.join(self.output_dir, "cloud.ply"))

        plan_img, P_plan = create_plan_view(
            map, g_plane, height=height, thickness=0.2, colored=True
        )
        plan_img.save(os.path.join(self.output_dir, "floor_plan.png"))
        # P_plan = np.array(P_plan).reshape(
        #    (-1, 4)
        # )

        np.savetxt(
            os.path.join(self.output_dir, "P_plan.txt"),
            P_plan,
            fmt="%f",
            delimiter=",",
        )
