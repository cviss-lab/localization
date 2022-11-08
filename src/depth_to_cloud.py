import open3d as o3d
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath
import json
from scipy.spatial.transform import Rotation

images_folder = '/home/zaid/datasets/22-11-03-ParkStBridge/images'

if os.path.exists(join(images_folder, 'K2.txt')):
    K2 = np.loadtxt(join(images_folder, 'K2.txt'))
elif os.path.exists(join(images_folder, 'intrinsics.json')):
    with open(join(images_folder, 'intrinsics.json')) as f:
        intrinsics = json.load(f)
    K2 = np.array(intrinsics['camera_matrix'])
else:
    raise Exception('No intrinsics file found')

fx = K2[0,0]
fy = K2[1,1]
cx = K2[0,2]
cy = K2[1,2]

img_id = 280

depth = cv2.imread(join(images_folder,'depth','{}.png'.format(img_id)),cv2.IMREAD_UNCHANGED)
color = cv2.cvtColor(cv2.imread(join(images_folder,'rgb','{}.png'.format(img_id))), cv2.COLOR_BGR2RGB)
height, width, _ = color.shape

depth = o3d.geometry.Image(np.float32(depth))
color = o3d.geometry.Image(color)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth,depth_trunc=100000, convert_rgb_to_intensity=False)


intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

poses = np.loadtxt(join(images_folder,'poses.csv'), delimiter=',')
pose = poses[img_id-1,:]
pose = pose.reshape(-1)
q = pose[4:]
ext = np.eye(4)
ext[:3,:3] = Rotation.from_quat(q).as_matrix()
ext[:3,3] = np.array(pose[1:4])
ext = np.linalg.inv(ext)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, ext)

o3d.io.write_point_cloud(join(images_folder, 'cloud.ply'),pcd)


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)
# vis.run()

