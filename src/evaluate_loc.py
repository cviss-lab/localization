#!/usr/bin/env python3
import shutil
import numpy as np
import os
from os.path import join, dirname, realpath
import cv2
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
import json
from ssloc_offline import ssloc
from render_depthmap import VisOpen3D

global img_i
global data_folder
img_i = 0

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

def check_pose_estimation(pcd_file, pose, Ip, view = 1, imshow=True, fsave=None, K2=None, panorama=False, marker_file=None,show_results=True):
    
    if panorama:
    
        I2_l, T2_l, K2 = utils.fun_rectify_views(Ip, 90)
        
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
        T_c2_m1 = utils.T_inv(T_m1_c2)


    if marker_file is not None:
        e_scaled = measure_error(I2_front,T_c2_m1,K2,marker_file,view,panorama)
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

def check_all_poses(pcd_file,query_folder,out_folder,panorama=False,marker_file=None,show_results=True,results_prefix=''):

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
    for fname in sorted(os.listdir(query_folder)):
        ext = fname.split('.')[-1]
        if ext == 'jpg' or ext == 'png' or ext == 'JPG':
            img_l.append(fname)

    poses = np.loadtxt(join(query_folder,results_prefix+'poses.csv'), delimiter=',')
    if panorama:
        views_l = [1,3,4]
    else:
        views_l = [1]

    error = []
    for img_id in range(poses.shape[0]):
        for view in views_l:
            pose = poses[img_id,:]
            Ip = cv2.imread(join(query_folder,'{}').format(img_l[img_id]))
            e_scaled = check_pose_estimation(pcd_file, pose, Ip, view=view, imshow=False, fsave=join(out_folder,'out{}_{}.jpg'.format(img_id,view)), panorama=panorama, 
                                  K2=K2, marker_file=marker_file,show_results=show_results)   
            error.append(e_scaled)

    return error

def measure_error(Ip, T_c2_m1,K2, marker_file, view, panorama, marker_size=20):
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
        marker_uv = utils.detect_markers(Ip,xy_array=True)
        if len(marker_uv) == 0:
            return -1
        marker_uv = marker_uv[0]['coords'].T    

    marker_3d = np.loadtxt(marker_file,delimiter=',')

    tvec, rvec = utils.matrix2poses(utils.T_inv(T_c2_m1))

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
    # data_folder = '/home/zaid/datasets/22-03-31-StructuresLabPanorama-processed'
    # data_folder = '/home/zaid/datasets/22-10-11-ParkStBridge-oneside-processed'
    data_folder = '/home/zaid/datasets/22-11-03-ParkStBridge'
    
    images_folder = join(data_folder, 'images')
    pcd_file = join(data_folder,'r3live_output/rgb_pt.pcd')
    query_panorama = join(data_folder,'panorama')
    marker_file = join(data_folder,'marker.txt')
    query_front = join(data_folder,'HL2')
    out_folder1 = join(data_folder,'out1')
    out_folder2 = join(data_folder,'out2')
    show_results=False

    # # create new anchors    
    # n = ssloc(debug=False, data_folder=images_folder, create_new_anchors=True)
    # n.create_offline_anchors()    
    # n.create_offline_anchors(skip=1, num_images=250)    

    # # label marker corners in panoramas
    # marker_uv_l = []
    # img_id = -1
    # for i,fname in enumerate(sorted(os.listdir(query_panorama))):
    #     ext = fname.split('.')[-1]
    #     if ext == 'jpg' or ext == 'png' or ext == 'JPG':
    #         views_l = [1,3,4]
    #         Ip = cv2.imread(join(query_panorama,fname), cv2.IMREAD_COLOR)
    #         #Ip = cv2.resize(Ip,(int(Ip.shape[1]/4),int(Ip.shape[0]/4))) # REMOVE
    #         I2_l, T2_l, K2 = utils.fun_rectify_views(Ip, 90)
    #         for view in views_l:
    #             marker_uv = utils.detect_markers_manual(I2_l[view-1])
    #             img_id += 1
    #             if len(marker_uv) > 0:
    #                 marker_uv = marker_uv[0]['coords'].T 
    #                 for u,v in marker_uv:
    #                     marker_uv_l.append([img_id,i,u,v])
    # np.savetxt( join(data_folder,'panorama','markers.csv'),marker_uv_l,delimiter=',', fmt='%i')

    # # query panromas
    error_all = []
    detectors = ['SuperPoint','SIFT','ORB','loftr']
    matchers = ['SuperGlue',None, None,'loftr']

    # # query panoramas
    for d,m in zip(detectors,matchers) :
        n = ssloc(data_folder=images_folder, create_new_anchors=False,detector=d, matcher=m)
        # ssl
        # n.query_panoramas(query_panorama, optimization=False, one_view=True, results_prefix=d+'_ssl_')
        # msl
        n.query_panoramas(query_panorama, optimization=False, one_view=False, results_prefix=d+'_msl_', max_reproj_error=10)


    # # panorama ssl error
    # for d,m in zip(detectors,matchers) :
    #     img_i = 0
    #     error = []
    #     print('ssl errors: '+d)
    #     error = check_all_poses(pcd_file,query_panorama,out_folder1,panorama=True,marker_file=marker_file,
    #                             show_results=show_results, results_prefix=d+'_ssl_')
    #     error_all.append(error)
    # # np.savetxt(join(data_folder,'panorama_error_ssl.csv'), error_all, delimiter=',')
    # with open(join(data_folder,'panorama_error_ssl.csv'), 'w') as f:
    #     for e in error_all:
    #         for ei in e:
    #             f.write(str(ei)+' ')
    #         f.write('\n')     

    # # panorama msl error
    error_all = []
    for d,m in zip(detectors,matchers) :
        img_i = 0
        error = []
        print('msl errors: '+d)
        error = check_all_poses(pcd_file,query_panorama,out_folder1,panorama=True,marker_file=marker_file,
                                show_results=show_results, results_prefix=d+'_msl_')
        error_all.append(error)
    # np.savetxt(join(data_folder,'panorama_error_msl.csv'), error_all, delimiter=',')
    with open(join(data_folder,'panorama_error_msl.csv'), 'w') as f:
        for e in error_all:
            for ei in e:
                f.write(str(ei)+' ')
            f.write('\n') 

    # # query front
    # n = ssloc(debug=False, data_folder=images_folder, create_new_anchors=False)
    # n.query_front(query_front, max_reproj_error=5)
    # n.query_front_multiple(query_front,max_reproj_error=5)
    # print('front camera errors:')
    # check_all_poses(pcd_file,query_front,out_folder2,panorama=False,marker_file=marker_file,show_results=show_results)

    # # save panorama poses in potree format
    # fposes2 = join(query_panorama,'poses_potree.txt')
    # poses2 = utils.poses_opencv2potree(poses)
    # with open(fposes2, 'w') as f:
    #     for i,fname in enumerate(img_l):           
    #         p = poses2[i,:]
    #         f.write(fname+'\t0\t'+'\t'.join([str(x) for x in p])+'\n')            


if __name__ == '__main__':
    main()
