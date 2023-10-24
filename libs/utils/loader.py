import contextlib
import os
import json
import abc
import typing

import PIL.Image
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import sys
try:
    import open3d as o3d
except:
    class o3d:
        class geometry:
            class PointCloud:
                pass
    pass


Poses = typing.Dict[int, typing.List[float]]
Intrinsics = typing.Tuple[npt.ArrayLike, int, int]
GroundPoints = typing.List[typing.Tuple[float, float, float]]


class Loader(abc.ABC):
    @contextlib.contextmanager
    @abc.abstractmethod
    def load_file(self, filename: str):
        yield ""

    @abc.abstractmethod
    def exists(self, filename: str) -> bool:
        pass

    @abc.abstractmethod
    def listdir(self, path: str) -> typing.List[str]:
        pass

    def load_P_plan(self, filename: str = "P_plan.txt"):
        """
        Read the P_plan matrix stored in P_plan.txt
        :param filename: name of the file to look for
        :return: returns list of tuples
        """
        if not self.exists(filename):
            raise FileNotFoundError
        with self.load_file(filename) as path:
            P_plan = np.loadtxt(path, delimiter=',')
            return P_plan
        

    def load_annotations(self, filename = 'annotations.json'):
        with self.load_file(filename) as path:
            with open(path) as f:
                annotations = json.load(f)
        return annotations["annotations"]


    def load_gnd_pts(self, filename: str = "picking_list.txt") -> GroundPoints:
        """
        Read the ground points csv usually stored in picking_list.txt
        :param filename: name of file to look for
        :return: returns list of tuples [(X, Y, Z), (X, Y, Z), (X, Y, Z)]
        """
        if not self.exists(filename):
            return [(1, 0, 0), (2, 0, 0), (1, 0, 1)]
        with self.load_file(filename) as path:
            df = pd.read_csv(path, header=None)
            pts_list = []
            # Need a min. of three points
            for i in range(0, 3):
                s = tuple(df.iloc[i][1:])
                pts_list.append(s)

            return pts_list

    def load_pc(self, filename: str = "cloud.ply") -> o3d.geometry.PointCloud:
        """
        Read the point cloud ply usually stored in cloud.ply
        :param filename: name of the file to look for
        :return: returns open3d point cloud object
        """
        if 'open3d' not in sys.modules:
            print('Open3D not installed')
            return None

        with self.load_file(filename) as path:
            return o3d.io.read_point_cloud(path, format="ply")

    def load_intrinsics(self, filename: str = "intrinsics.json") -> Intrinsics:
        """
        :param filepath: directory/intrinsics.json
        :return: K, w, h
        """
        with self.load_file(filename) as path:
            with open(path) as f:
                intrinsics = json.load(f)
            K = np.array(intrinsics["camera_matrix"])
            w = intrinsics["width"]
            h = intrinsics["height"]

            return K, w, h

    def load_poses(self, filename: str = "poses.csv") -> Poses:
        """
        :param filepath: directory/poses.csv
        :return: dictionary of poses
        """
        with self.load_file(filename) as path:
            df = pd.read_csv(path, header=None, index_col=0)
            poses = df.to_dict(orient="index")
            for p in poses.keys():
                poses[p] = list(poses[p].values())
            return poses

    def load_imgs_dict(self, poses: Poses, image_dir: str):
        """
        :param imgs_folder: directory/rgb
        :param poses: dictionary of poses
        :return: dictionary of images
        """
        imgs_dict = dict()
        files = self.listdir(image_dir)
        if len(files) > 0:
            id = lambda filename: int(filename.split(".")[0])

            imgs_dict = {
                id(filename): filename for filename in files if id(filename) in poses
            }

        return imgs_dict

    def load_image(self, filename: str) -> PIL.Image.Image:
        with self.load_file(filename) as path:
            return PIL.Image.open(path)

    def load_depth(self, filename: str) -> cv2.Mat:
        with self.load_file(filename) as path:
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)


class LocalLoader(Loader):
    working_dir: str

    def __init__(self, dir: str):
        self.working_dir = dir

    @contextlib.contextmanager
    def load_file(self, filename: str):
        yield os.path.join(self.working_dir, filename)

    def exists(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.working_dir, filename))

    def listdir(self, path: str) -> typing.List[str]:
        return os.listdir(os.path.join(self.working_dir, path))
