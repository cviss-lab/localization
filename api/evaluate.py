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
import json

global img_i
global data_folder
img_i = 0

def load_localizer(data_folder):
     
    if not os.path.exists(data_folder):
        return None
    l = LocalLoader(data_folder)
    loc = localization.Localizer(l)

    return loc

def query_front_multiple(query_data_folder, localizer, out_folder, base_link=False, retrieved_anchors=5, 
                         frame_rate=30, sample_rate=1, sliding_window = 5, min_matches=50):

        shutil.rmtree(out_folder,ignore_errors=True)
        os.makedirs(out_folder,exist_ok=True)

        skip_rate = int(frame_rate/sample_rate)

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
        
        poses_all = np.loadtxt(join(query_data_folder, 'poses.csv'), delimiter=',')
        poses_l = []
        id_l = []   
        poses_out = []  

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):

            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            id = int(filename.split('.')[0])

            if id % skip_rate != 0:
                continue

            # if id not in [1,100,500,1000]:
            #     continue            

            id_l.append(id)

            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)
            I2_l.append(I2)

            pose = poses_all[id-1,1:]            

            if base_link:                
                T_m2_c2 = pose2matrix(pose)
                T_link_c2 = pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
                T_m2_link = T_m2_c2
                T_m2_c2 = T_m2_link.dot(T_link_c2)
                pose = matrix2pose(T_m2_c2)

            poses_l.append(pose.tolist())

            if len(poses_l) == sliding_window:
                T_m1_m2, inliers = localizer.callback_query_multiple(I2_l, poses_l, K2, retrieved_anchors=retrieved_anchors, min_matches=min_matches)
        
                if T_m1_m2 is not None:                  

                    for pose,id in zip(poses_l,id_l):               

                        T_m2_c2 = pose2matrix(pose)
                        T_m1_c2 = T_m1_m2.dot(T_m2_c2)

                        R = Rotation.from_matrix(T_m1_c2[:3, :3])
                        q = R.as_quat()
                        t = T_m1_c2[:3, 3].T
                        pose = np.concatenate(([id],t, q), axis=0)
                        poses_out.append(pose)
                else:
                    print('localization failed!')                    

                I2_l = []
                poses_l = []
                id_l = []

        poses_out = np.array(poses_out)

        np.savetxt(join(out_folder, 'localized_poses.csv'), poses_out, delimiter=',')
        np.savetxt(join(out_folder, 'num_inliers.csv'), inliers, delimiter=',', fmt='%i')


def query_front_single(query_data_folder, localizer, out_folder, retrieved_anchors=5, frame_rate=30, sample_rate=1, min_matches=50):

        shutil.rmtree(out_folder,ignore_errors=True)
        os.makedirs(out_folder,exist_ok=True)

        skip_rate = int(frame_rate/sample_rate)

        query_img_dir = join(query_data_folder, 'rgb')

        if os.path.exists(join(query_data_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_data_folder, 'K2.txt'))
        elif os.path.exists(join(query_data_folder, 'intrinsics.json')):
            with open(join(query_data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')  

        poses_out = np.empty((0, 8))
        inliers_l = []

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):

            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            id = int(filename.split('.')[0])

            if id % skip_rate != 0:
                continue

            # if id not in [1,100,500,1000]:
            #     continue                

            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)

            T_m1_c2, inliers = localizer.callback_query(I2, K2, retrieved_anchors=retrieved_anchors, min_matches=min_matches) 

            if T_m1_c2 is None:
                T_m1_c2 = np.eye(4)  
                inliers_l.append(0)
            else:
                inliers_l.append(inliers)

            R = Rotation.from_matrix(T_m1_c2[:3, :3])
            q = R.as_quat()
            t = T_m1_c2[:3, 3].T
            pose = np.concatenate(([id],t, q), axis=0)
            poses_out = np.vstack((poses_out, pose))

        np.savetxt(join(out_folder, 'localized_poses.csv'), poses_out, delimiter=',')
        np.savetxt(join(out_folder, 'num_inliers.csv'), inliers_l, delimiter=',', fmt='%i')


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

def check_pose_estimation(pcd_file, pose, Ip, view = 1, imshow=True, fsave=None, K2=None, panorama=False,show_results=True, marker_size=20):
    
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


    e_scaled = measure_error(I2_front,T_c2_m1,K2,panorama,marker_size)

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

def check_all_poses(pcd_file,query_folder, out_folder,panorama=False,show_results=True,results_prefix='',marker_size=20):
    global img_i
    img_i = 0

    query_imgs_folder=query_folder+'/rgb'

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
    ext_type = 'jpg'
    for fname in sorted(os.listdir(query_imgs_folder)):
        ext = fname.split('.')[-1]
        if ext == 'jpg' or ext == 'png' or ext == 'JPG':
            ext_type = ext

    poses = np.loadtxt(join(out_folder,results_prefix+'localized_poses.csv'), delimiter=',')
    poses_dict = {int(pose[0]):pose[1:] for pose in poses}

    if panorama:
        views_l = [1,3,4]
    else:
        views_l = [1]

    error = []
    for img_id in poses_dict.keys():
        for view in views_l:
            pose = poses_dict[img_id]
            Ip = cv2.imread(join(query_imgs_folder,f'{img_id}.{ext_type}'))
            e_scaled = check_pose_estimation(pcd_file, pose, Ip, view=view, imshow=False, fsave=join(out_folder,'out{}_{}.jpg'.format(img_id,view)), panorama=panorama, 
                                  K2=K2,show_results=show_results, marker_size=marker_size)   
            error.append(e_scaled)

    return error

def measure_error(Ip, T_c2_m1,K2, panorama, marker_size):
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
        marker = detect_markers(Ip,xy_array=True)
        if len(marker) == 0:
            return -1
        marker_uv = marker[0]['coords'].T    
        marker_id = marker[0]['id']

    # read json file to get marker 3d coordinates from marker id
    marker_file = join(data_folder,'markers_dict.json')
    with open(marker_file) as f:
        markers = json.load(f)

    if marker_id not in markers:
        return -1

    marker_3d = markers[marker_id]
    marker_3d = np.array(marker_3d)
    marker_3d = np.array(marker_3d,dtype=np.float32)

    tvec, rvec = matrix2poses(T_inv(T_c2_m1))

    marker_uv_proj = cv2.projectPoints(marker_3d,rvec,tvec,K2,None)[0].reshape(-1,2)

    e = marker_uv.mean(axis=0) - marker_uv_proj.mean(axis=0)
    e = np.linalg.norm(e)

    side = np.linalg.norm(marker_uv[0,:] - marker_uv[1,:])
    e_scaled = marker_size/side * e    
    # print('{:3f}'.format(e_scaled))
    
    return e_scaled

def find_markers_in_anchors(localizer, frame_rate=10, sampling_rate=1):

    w = localizer.image_width
    h = localizer.image_height
    K1 = localizer.camera_matrix 

    markers_dict = dict()

    skip_rate = int(frame_rate/sampling_rate)

    for img_idx in localizer.poses.keys():
        
        if img_idx % skip_rate != 0:
            continue

        I1 = localizer.load_rgb(img_idx)
        D1 = localizer.load_depth(img_idx)
        pose1 = localizer.get_pose(img_idx)

        # detect markers
        markers = detect_markers(I1,xy_array=True)
        if len(markers) == 0:
            continue
        marker_id = markers[0]['id']
        marker_uv = markers[0]['coords'].T           
        marker_3d, valid_ind = project_2d_to_3d(marker_uv.T, D1, K1, pose1, w, h, return_valid_ind=True)     
        
        if len(valid_ind) == 0:
            continue

        if marker_3d.shape[1] != 4:
            continue

        markers_dict[marker_id] = marker_3d.tolist()
           
    ## save markers_dict to json file
    with open(join(data_folder,'markers_dict.json'), 'w') as fp:
        json.dump(markers_dict, fp)
        
def evaluate_localization(data_folder, query_folder, out_folder, marker_size, multiple=False, run_relocalization=True,
                          retrieved_anchors=10, frame_rate=30, sample_rate=0.5):  

    pcd_file = join(data_folder,'cloud.pcd')

    # load localizer
    n = load_localizer(data_folder)    

    # find markers in anchors
    find_markers_in_anchors(n)  

    # localize query images
    if multiple:
        if run_relocalization:
            query_front_multiple(query_folder,n,out_folder,base_link=True, retrieved_anchors=retrieved_anchors, 
                                frame_rate=frame_rate, sample_rate=sample_rate, min_matches=1)
    else:
        if run_relocalization:
            query_front_single(query_folder,n,out_folder, retrieved_anchors=retrieved_anchors, 
                                frame_rate=frame_rate, sample_rate=sample_rate, min_matches=1)    

    # evaluate error
    error1 = check_all_poses(pcd_file,query_folder,out_folder,panorama=False,
                            show_results=True,marker_size=marker_size)
    with open(join(out_folder,'error.csv'), 'w') as f:
        for ei in error1:
            f.write(str(ei)+' ')
        f.write('\n') 


def main():
    global data_folder

    # data_folder = '/home/zaid/datasets/iwshm2023_data/project_1'
    data_folder = '/home/zaid/datasets/iwshm2023_data/iphone/c034beb4bb_out'
    query_folder = '/home/zaid/datasets/iwshm2023_data/drone'
    marker_size = 16.51 # cm

    out_folder1 = join(query_folder,'out1')
    out_folder2 = join(query_folder,'out2')

    # evaluate ssl
    evaluate_localization(data_folder, query_folder, out_folder2, marker_size, multiple=False, 
                            retrieved_anchors=10, frame_rate=30, sample_rate=5)

    # evaluate msl
    # evaluate_localization(data_folder, query_folder, out_folder1, marker_size, multiple=True, 
    #                         retrieved_anchors=10, frame_rate=30, sample_rate=1)    

if __name__ == '__main__':
    main()
