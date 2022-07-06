#!/usr/bin/env python3  
from pickle import FALSE
import numpy as np
import sys
import os
from os.path import join, dirname, realpath
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time
import utils
from Equirec2Perspec import Equirectangular

class Node:

    def __init__(self):
        self.kp1 = []
        self.des1 = np.empty((0,128),dtype=np.float32)
        self.pts3D = np.empty((0,3),dtype=np.float32)

    def create_anchor(self,I1,D1,K1,pose1):

        kp1, des1 = self.feature_detection(I1)
        pts1 = np.float32([ kp.pt for kp in kp1 ])
        x_c,y_c,z_c = utils.project_2d_to_3d(pts1.T,K1,D1)
        pts3D_c = np.array([x_c,y_c,z_c,np.ones(x_c.shape[0])])

        tc = pose1[:3]
        qc = pose1[3:]
        
        Tc = np.eye(4)
        R = Rotation.from_quat(qc)
        Tc[:3,:3] = R.as_matrix()
        Tc[:3,3] = tc

        pts3D = Tc.dot(pts3D_c)
        pts3D = pts3D[:3,:]/pts3D[3,:]

        pts3D = pts3D.T  

        valid = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
        pts3D = pts3D[valid]
        kp1 = [kp1[v] for v in valid]
        des1 = des1[valid,:]    

        self.kp1 += kp1
        self.des1 = np.vstack([self.des1,des1])
        self.pts3D = np.vstack([self.pts3D,pts3D])    


    def callback_query_panorama(self,I2):
        print('query image recieved!')

        FOV=90
        E2P = Equirectangular(I2)
        I2_front,K2 = E2P.GetPerspective(FOV,FOV,0,0)
        I2_back,K2 = E2P.GetPerspective(FOV,FOV,180,0)
        I2_l =[I2_front,I2_back]
        q_poses_l = [[0,0,0,-0.5,0.5,-0.5,0.5],
                [0,0,0,0.5,0.5,-0.5,-0.5]]

        pts2D_l = []
        pts3D_l = []
        pts_idx = []
        tvecs_l = []
        rvecs_l = []

        for I2 in I2_l:

            kp2, des2 = self.feature_detection(I2)
            
            matches = self.feature_matching(self.des1,des2)            
            # pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])            
            pts2D = np.float32([ kp2[m.trainIdx].pt for m in matches ])

            m_query_idx = [m.queryIdx for m in matches]
            pts3D_m = self.pts3D[m_query_idx,:]

            retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D_m, pts2D, K2, None,flags=cv2.SOLVEPNP_P3P)
            e1=pnp_error(pts3D_m,pts2D,rvecs,tvecs,K2); e1=np.array(e1,dtype=int)
            # inliers = e1 < 100
            # if retval:
            #     pts2D_l.append(pts2D[inliers,:])
            #     pts3D_l.append(pts3D_m[inliers,:])            
            #     pts_idx.append(np.array(m_query_idx)[inliers].reshape(-1))
            #     rvecs_l.append(rvecs)
            #     tvecs_l.append(tvecs)
            
            pts2D_l.append(pts2D)
            pts3D_l.append(pts3D_m)            
            pts_idx.append(np.array(m_query_idx).reshape(-1))
            rvecs_l.append(rvecs)
            tvecs_l.append(tvecs)            

        # utils.save_variable(pts3D,'pts3D.txt')
        # utils.save_variable(pts2D_l,'pts2D_l.txt')
        # utils.save_variable(pts_idx,'pts_idx.txt')
        # utils.save_variable(tvecs_l,'tvecs_l.txt')
        # utils.save_variable(rvecs_l,'rvecs_l.txt')
        
        T_m1_m2,tvecs_l,rvecs_l,best_inlier_idxs=multiviewSolvePnPRansac(self.pts3D, pts_idx, pts2D_l, q_poses_l, K2) 

        pts2D_in  = [[] for _ in range(len(pts2D_l))]
        pts_idx_in  = [[] for _ in range(len(pts_idx))]
        for i,inliers in enumerate(best_inlier_idxs):
            pts2D_in[i] = pts2D_l[i][inliers,:]
            pts_idx_in[i] = pts_idx[i][inliers]

        tvecs_l, rvecs_l = apply_bundle_adjustment(self.pts3D,pts_idx_in,pts2D_in,tvecs_l,rvecs_l,K2,plot=True)

        best_inlier2_idxs = best_inlier_idxs
        for i,(rvecs,tvecs) in enumerate(zip(rvecs_l,tvecs_l)):
            e2 = pnp_error(self.pts3D[pts_idx[i],:],pts2D_l[i],rvecs,tvecs,K2)            
            best_inlier2_idxs[i] = np.where([e2 < 50])[1]

        pass
        # e2=pnp_error(pts3D,pts2D,rvecs2,tvecs2,K2); e2=np.array(e2,dtype=int)
  

    def draw_matches(self,I1,I2,kp1,kp2,matches):
        
        img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
        self.debug_img = img
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()

    def feature_detection(self,I,detector='SIFT',fname=None):
        
        if detector == 'SIFT':
            gray= cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(gray,None)
        elif detector == 'ORB':
            orb = cv2.ORB_create(nfeatures=5000)
            kp, des = orb.detectAndCompute(I,None)
        elif detector == 'SURF':
            surf = cv2.xfeatures2d.SURF_create()
            kp, des = surf.detectAndCompute(I,None)
        else:
            if os.path.exists(fname):
                os.remove(fname)
            kp = detect_features.main(I,fname,detector,model=self.detector_model)  
            des = None        

        return kp, des

    def feature_matching(self,des1,des2,detector='SIFT',matcher=None,fname1=None,fname2=None):

        if detector == 'ORB' or detector == 'SIFT' or detector == 'SURF':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1,des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            matches = good
            return matches     
        else:
            matches = match_features.main(fname1,fname2,detector,matcher,model=self.matcher_model)

        return matches    
        
def my_solvePnPRansac(pts3D, pts2D, K2, max_reproj_error=8, max_iterations=1000):

    rvecs = None
    tvecs = None
    best_inlier_idxs = []

    n = 4
    n_data = pts2D.shape[0]
    iterations = 0
    best_inlier_idxs = []
    while iterations < max_iterations:
        test_idxs = np.random.randint(0,n_data,n)
        pts2D_test = pts2D[test_idxs,:]
        pts3D_test = pts3D[test_idxs,:]
        retval,rvecs_test,tvecs_test = cv2.solvePnP(pts3D_test,pts2D_test,K2,None,flags=cv2.SOLVEPNP_P3P)
        if not retval:
            continue
        test_err = pnp_error(pts3D,pts2D,rvecs_test,tvecs_test,K2)
        inlier_idxs = np.where([test_err < max_reproj_error])[1] # select indices of rows with accepted points

        if len(inlier_idxs) > len(best_inlier_idxs):
            tvecs = tvecs_test
            rvecs = rvecs_test
            best_inlier_idxs = inlier_idxs
        iterations+=1

    return rvecs,tvecs,best_inlier_idxs

def multiviewSolvePnPRansac(pts3D, pts_idx, pts2D_l, poses_l, K2, max_reproj_error=25, max_iterations=1000):

    n_test = 4
    n_imgs = len(pts2D_l)
    iterations = 0
    T_m1_m2 = None
    if len(poses_l[0])>7:
        k0 = len(poses_l[0])-6
    else:
        k0 = 0
    T_m2_c_l = [utils.pose2matrix(pose[k0:]) for pose in poses_l]
    best_inlier_idxs = [[] for _ in range(n_imgs)]  
    best_err = None
    # for i_img in range(n_imgs):
    iterations = 0
    best_inlier_idxs = [[] for _ in range(n_imgs)]  
    while iterations < max_iterations:
        i_img = np.random.randint(0,n_imgs)
        n_data = pts2D_l[i_img].shape[0]
        test_idxs = np.random.randint(0,n_data,n_test)
        pts2D_test = pts2D_l[i_img][test_idxs,:]
        pts3D_test = pts3D[pts_idx[i_img]][test_idxs,:]
        retval,rvecs_test,tvecs_test = cv2.solvePnP(pts3D_test,pts2D_test,K2,None,flags=cv2.SOLVEPNP_P3P)
        if not retval:
            continue
        T_m1_c_test = utils.poses2matrix(tvecs_test,rvecs_test)
        T_m2_c = T_m2_c_l[i_img]
        T_c_m2 = utils.T_inv(T_m2_c)
        T_m1_m2_test = np.dot(T_m1_c_test,T_c_m2)

        test_err = multiview_pnp_error(pts3D, pts_idx, pts2D_l, K2, T_m1_m2_test, T_m2_c_l)
        inlier_idxs = [np.where([e < max_reproj_error])[1].tolist() for e in test_err] # select indices of rows with accepted points

        if utils.len_subelems(inlier_idxs) > utils.len_subelems(best_inlier_idxs):
            T_m1_m2 = T_m1_m2_test
            best_inlier_idxs = inlier_idxs
            best_err = [np.array(e,dtype=np.int) for e in test_err]
            # print(T_m1_m2[:3,3])
        iterations+=1
        
    tvecs_l = [utils.matrix2poses(T_m1_m2.dot(T_m2_c))[0] for T_m2_c in T_m2_c_l]
    rvecs_l = [utils.matrix2poses(T_m1_m2.dot(T_m2_c))[1] for T_m2_c in T_m2_c_l]

    return T_m1_m2,tvecs_l,rvecs_l,best_inlier_idxs

def apply_bundle_adjustment(pts3D,pts_idx,pts2D_l,tvecs_l,rvecs_l,K,plot=False):

    from bundle_adjustment import BundleAdjustment2

    bundle_adjustment = BundleAdjustment2(optimize_points=False,attach_cameras=True)

    bundle_adjustment.read_from_data(pts3D,pts_idx,pts2D_l,tvecs_l,rvecs_l,K)

    n_cameras = bundle_adjustment.camera_params.shape[0]
    n_points = bundle_adjustment.points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * bundle_adjustment.points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    _,f0 = bundle_adjustment.fun_init()

    if plot:
        bundle_adjustment.init_viz()
        bundle_adjustment.add_viz('c')
        
    bundle_adjustment.bundle_adjustment_sparsity()
    t0 = time.time()

    res = bundle_adjustment.least_squares()
    t1 = time.time()    

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    if plot:
        bundle_adjustment.add_viz('r')
        plt.show()

        plt.plot(f0)
        plt.plot(res.fun)
        
        plt.show()  

    return bundle_adjustment.tvecs, bundle_adjustment.rvecs  

def pnp_error(pts3D,pts2D,rvecs,tvecs,K2):
    pts2D_reproj = cv2.projectPoints(pts3D,rvecs,tvecs,K2,None)[0].reshape(-1,2)
    e = np.linalg.norm(pts2D-pts2D_reproj,axis=1)    
    return e

def multiview_pnp_error(pts3D, pts_idx, pts2D_l, K2, T_m1_m2_test, T_m2_c_l):
    n = len(pts2D_l)
    e_l = [None for _ in range(n)]
    for i in range(n):
        pts3D_i = pts3D[pts_idx[i],:]
        pts2D_i = pts2D_l[i]
        T_m1_ci = T_m1_m2_test.dot(T_m2_c_l[i])
        tvecs,rvecs = utils.matrix2poses(T_m1_ci)
        e_l[i] = pnp_error(pts3D_i,pts2D_i,rvecs,tvecs,K2)
    return e_l



def main_from_data():
    import glob
    import re

    D1 = cv2.imread('robot/depth_202.png',cv2.IMREAD_UNCHANGED)
    I1 = cv2.imread('robot/rgb_202.png')
    K1 = np.loadtxt('robot/K1.txt')

    K2 = np.loadtxt('HL2/K2.txt')
    I2 = cv2.imread('HL2/50.jpg')    
    I2_flist = glob.glob('HL2/*.jpg')    
    I2_names = [int(f.split('\\')[1].split('.')[0]) for f in I2_flist]
    I2_l = [cv2.imread(f) for f in I2_flist]
    poses_l = np.loadtxt('HL2/poses.csv',delimiter=',')
    poses_l = poses_l[[p in I2_names for p in poses_l[:,0]],:]
    
    n=Node(I1,D1,K1)
    n.callback_query(I2_l,poses_l,K2)     

def main_panorama():

    K1 = np.loadtxt('data/handheld/K1.txt')
    I1_l = utils.read_images_folder('data/handheld/rgb/*.png')
    D1_l = utils.read_images_folder('data/handheld/depth/*.png',cv2_flags=cv2.IMREAD_UNCHANGED)
    poses_l = utils.read_txt_folder('data/handheld/poses/*.txt')
    
    I2 = cv2.imread('data/panorama/panorama.jpg')
    
    n=Node()
    for I1,D1,pose1 in zip(I1_l,D1_l,poses_l):        
        n.create_anchor(I1,D1,K1,pose1)
    n.callback_query_panorama(I2) 

def main_from_saved():  
    #save variables here
    pts3D =   utils.load_variable('pts3D.txt')
    pts2D_l = utils.load_variable('pts2D_l.txt')
    pts_idx = utils.load_variable('pts_idx.txt')
    tvecs_l = utils.load_variable('tvecs_l.txt')
    rvecs_l = utils.load_variable('rvecs_l.txt')
    K2 = np.loadtxt('HL2/K2.txt')     
    apply_bundle_adjustment(pts3D,pts_idx,pts2D_l,tvecs_l,rvecs_l,K2)
    # multiviewSolvePnPRansac(pts3D, pts_idx, pts2D_l, poses_l, K2)

if __name__ == '__main__':
    # main_from_saved()   
    # main_from_data()
    main_panorama()        
    