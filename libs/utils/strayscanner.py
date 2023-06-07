import cv2
import open3d as o3d
import numpy as np
from PIL import Image


def load_depth(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
        depth_mm[confidence < filter_level] = 0
    return o3d.geometry.Image(depth_m), depth_mm

def load_confidence(path):
    return np.array(Image.open(path))

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