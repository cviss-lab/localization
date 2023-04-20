#!/usr/bin/env python3
import shutil
import numpy as np
import os
from os.path import join, dirname, realpath
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
import json
import localization
from libs.utils.render_depthmap import VisOpen3D
from libs.utils.projection import *
from libs.utils.loader import *
from libs.utils.utils import *

global img_i
global data_folder
img_i = 0

def load_localizer(data_folder):
     
    if not os.path.exists(data_folder):
        return None
    l = LocalLoader(data_folder)
    loc = localization.Localizer(l)

    return loc

def query_front_multiple(query_data_folder, localizer, base_link=False, retrieved_anchors=5):

        query_img_dir = join(query_data_folder, 'rgb')

        if os.path.exists(join(query_data_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_data_folder, 'K2.txt'))
        elif os.path.exists(join(query_data_folder, 'intrinsics.json')):
            with open(join(query_data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')  

        I2_l = []
        poses_l = np.loadtxt(join(query_data_folder, 'poses.csv'), delimiter=',')

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            id = int(filename.split('.')[0])
            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)
            I2_l.append(I2)

            if base_link:
                pose = poses_l[id-1,1:]
                T_m2_c2 = pose2matrix(pose)
                T_link_c2 = pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
                T_m2_link = T_m2_c2
                T_m2_c2 = T_m2_link.dot(T_link_c2)
                pose = matrix2pose(T_m2_c2)
                poses_l[id-1,1:] = pose
        poses_l = [p[1:].tolist() for p in poses_l]

        T_m1_m2, inliers = localizer.callback_query_multiple(I2_l, poses_l, K2, retrieved_anchors=retrieved_anchors)
        
        if T_m1_m2 is None:
            print('localization failed!')
            return

        poses = np.empty((0, 7))
        for pose in poses_l:

            T_m2_c2 = pose2matrix(pose)
            T_m1_c2 = T_m1_m2.dot(T_m2_c2)

            R = Rotation.from_matrix(T_m1_c2[:3, :3])
            q = R.as_quat()
            t = T_m1_c2[:3, 3].T
            pose = np.concatenate((t, q), axis=0)
            poses = np.vstack((poses, pose))

        np.savetxt(join(query_data_folder, 'localized_poses.csv'), poses, delimiter=',')
        np.savetxt(join(query_data_folder, 'num_inliers.csv'), inliers, delimiter=',', fmt='%i')


def query_front_single(query_data_folder, localizer, retrieved_anchors=5):

        query_img_dir = join(query_data_folder, 'rgb')

        if os.path.exists(join(query_data_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_data_folder, 'K2.txt'))
        elif os.path.exists(join(query_data_folder, 'intrinsics.json')):
            with open(join(query_data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')  

        poses = np.empty((0, 7))
        inliers_l = []

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            id = int(filename.split('.')[0])
            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)

            T_m1_c2, inliers = localizer.callback_query(I2, K2, retrieved_anchors=retrieved_anchors) 

            if T_m1_c2 is None:
                T_m1_c2 = np.eye(4)  
                inliers_l.append(0)
            else:
                inliers_l.append(inliers)

            R = Rotation.from_matrix(T_m1_c2[:3, :3])
            q = R.as_quat()
            t = T_m1_c2[:3, 3].T
            pose = np.concatenate((t, q), axis=0)
            poses = np.vstack((poses, pose))



        np.savetxt(join(query_data_folder, 'localized_poses.csv'), poses, delimiter=',')
        np.savetxt(join(query_data_folder, 'num_inliers.csv'), inliers_l, delimiter=',', fmt='%i')


def project_to_image(pcd_file,T_c1_m1,K,w,h,vis=None):

    if vis is None:
        vis = VisOpen3D(width=w, height=h, visible=False)
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd = pcd.voxel_down_sample(0.01)
        vis.add_geometry(pcd)

    vis.update_view_point(K, T_c1_m1)

    img = vis.capture_screen_float_buffer(show=False)
    img = np.uint8(255*np.array(img))
    img = np.dstack([img[:,:,2],img[:,:,1],img[:,:,0]])
    
    cx = K[0,2]
    cy = K[1,2]
    tx = cx-w/2
    ty = cy-h/2
    M = np.array([[1,0,tx],
                  [0,1,ty]])
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

    return img, vis

def check_pose_estimation(pcd_file, pose, Ip, view = 1, imshow=True, fsave=None, K2=None, panorama=False, marker_file=None,show_results=True, marker_size=20):
    
    if panorama:
    
        I2_l, T2_l, K2 = fun_rectify_views(Ip, 90)
        
        I2_front = I2_l[view-1]
        T_c1_c2 = T2_l[view-1]
    
        T_m1_c1 = np.eye(4)
        T_m1_c1[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        T_m1_c1[:3, 3] = pose[:3]
        T_c1_m1 = np.linalg.inv(T_m1_c1)
        T_c2_c1 = np.linalg.inv(T_c1_c2)
        T_c2_m1 = np.matmul(T_c2_c1, T_c1_m1)

    else:
        I2_front = Ip
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        T_m1_c2[:3, 3] = pose[:3]
        T_c2_m1 = T_inv(T_m1_c2)


    if marker_file is not None:
        e_scaled = measure_error(I2_front,T_c2_m1,K2,marker_file,view,panorama,marker_size)
    else:
        e_scaled = -1

    if show_results:
        I2_pcd,_ = project_to_image(pcd_file, T_c2_m1, K2, I2_front.shape[1], I2_front.shape[0])

        alpha_slider_max = 100
        title_window = 'Linear Blend'
        def on_trackbar(val):
            alpha = val / alpha_slider_max
            beta = ( 1.0 - alpha )
            dst = cv2.addWeighted(I2_pcd, alpha, I2_front, beta, 0.0)
            r = 0.5
            dst = cv2.resize(dst, (int(dst.shape[1]*r), int(dst.shape[0] * r)))
            if fsave is not None:
                cv2.imwrite(fsave.replace('.jpg','_{}.jpg'.format(val)), dst)
            if imshow:
                cv2.imshow(title_window, dst)

        if imshow:
            cv2.namedWindow(title_window)
            trackbar_name = 'Alpha x %d' % alpha_slider_max
            cv2.createTrackbar(trackbar_name, title_window , 0, alpha_slider_max, on_trackbar)
        # Show some stuff
        on_trackbar(0)
        on_trackbar(50)
        on_trackbar(100)
        # Wait until user press some key
        if imshow:
            cv2.waitKey() 

    return e_scaled

def check_all_poses(pcd_file,query_folder, out_folder,panorama=False,marker_file=None,show_results=True,results_prefix='',marker_size=20):

    query_imgs_folder=query_folder+'/rgb'

    if show_results:
        shutil.rmtree(out_folder,ignore_errors=True)
        os.makedirs(out_folder,exist_ok=True)

    if not panorama:
        if os.path.exists(join(query_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_folder, 'K2.txt'))
        elif os.path.exists(join(query_folder, 'intrinsics.json')):
            with open(join(query_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')
    else:
        K2 = None

    img_l = []
    for fname in sorted(os.listdir(query_imgs_folder)):
        ext = fname.split('.')[-1]
        if ext == 'jpg' or ext == 'png' or ext == 'JPG':
            img_l.append(fname)

    poses = np.loadtxt(join(query_folder,results_prefix+'localized_poses.csv'), delimiter=',')
    if panorama:
        views_l = [1,3,4]
    else:
        views_l = [1]

    error = []
    for img_id in range(poses.shape[0]):
        for view in views_l:
            pose = poses[img_id,:]
            Ip = cv2.imread(join(query_imgs_folder,'{}').format(img_l[img_id]))
            e_scaled = check_pose_estimation(pcd_file, pose, Ip, view=view, imshow=False, fsave=join(out_folder,'out{}_{}.jpg'.format(img_id,view)), panorama=panorama, 
                                  K2=K2, marker_file=marker_file,show_results=show_results, marker_size=marker_size)   
            error.append(e_scaled)

    return error

def measure_error(Ip, T_c2_m1,K2, marker_file, view, panorama, marker_size):
    global img_i
    global data_folder
    if panorama:
        
        marker = join(data_folder,'panorama','markers.csv')
        marker_uv = np.loadtxt(marker,delimiter=',')
        marker_uv = marker_uv[marker_uv[:,0]==img_i,1:]

        img_i += 1

        if len(marker_uv) == 0:
            return -1     
    else:
        marker_uv = detect_markers(Ip,xy_array=True)
        if len(marker_uv) == 0:
            return -1
        marker_uv = marker_uv[0]['coords'].T    

    marker_3d = np.loadtxt(marker_file,delimiter=',')
    marker_3d = marker_3d[:,1:]
    marker_3d = np.array(marker_3d,dtype=np.float32)

    tvec, rvec = matrix2poses(T_inv(T_c2_m1))

    marker_uv_proj = cv2.projectPoints(marker_3d,rvec,tvec,K2,None)[0].reshape(-1,2)

    e = marker_uv.mean(axis=0) - marker_uv_proj.mean(axis=0)
    e = np.linalg.norm(e)

    side = np.linalg.norm(marker_uv[0,:] - marker_uv[1,:])
    e_scaled = marker_size/side * e    
    # print('{:3f}'.format(e_scaled))
    
    return e_scaled

def main():
    global img_i
    global data_folder

    # data_folder = '/home/zaid/datasets/iwshm2023_data/project_1'
    data_folder = '/home/zaid/datasets/iwshm2023_data/iphone/c034beb4bb_out'
    query_front = '/home/zaid/datasets/iwshm2023_data/query'
    marker_size = 16.51 # cm

    pcd_file = join(data_folder,'cloud.pcd')
    marker_file = join(data_folder,'picking_list.txt')
    out_folder1 = join(data_folder,'out1')
    show_results=True

    # # localize query images
    n = load_localizer(data_folder)
    query_front_single(query_front,n, retrieved_anchors=20)
    # query_front_multiple(query_front,n,base_link=True, retrieved_anchors=20)

    # # panorama msl error
    error_all = []
    img_i = 0
    error = []
    # print('msl errors: '+d)
    error = check_all_poses(pcd_file,query_front,out_folder1,panorama=False,marker_file=marker_file,
                            show_results=show_results,marker_size=marker_size)
    error_all.append(error)
    with open(join(data_folder,'error_msl.csv'), 'w') as f:
        for e in error_all:
            for ei in e:
                f.write(str(ei)+' ')
            f.write('\n') 


if __name__ == '__main__':
    main()
