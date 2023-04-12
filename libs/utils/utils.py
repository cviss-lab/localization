import os
import shutil
import cv2
# import cv2.aruco as aruco
import numpy as np
#import scipy
from geometry_msgs.msg import Transform, TransformStamped, PoseStamped
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .Equirec2Perspec import *

def T_inv(Tmat):
    if Tmat.shape == (4,4):
        R = Tmat[:3,:3]
        t = Tmat[:3,3]
        R_inv = R.T
        t_inv = -R_inv.dot(t)     
        Tmat_inv = np.eye(4)
        Tmat_inv[:3,:3] = R_inv
        Tmat_inv[:3,3] = t_inv
        return Tmat_inv

def multiviewSolvePnPRansac(pts3D_l, pts2D_l, poses_l, K2, max_reproj_error=25, max_iterations=1000):

    n_test = 4
    n_imgs = len(pts2D_l)
    iterations = 0
    T_m1_m2 = None
    if len(poses_l[0])>7:
        k0 = len(poses_l[0])-6
    else:
        k0 = 0
    T_m2_c_l = [pose2matrix(pose[k0:]) for pose in poses_l]
    best_inlier_idxs = [[] for _ in range(n_imgs)]  
    best_err = None
    # for i_img in range(n_imgs):
    iterations = 0
    best_inlier_idxs = [[] for _ in range(n_imgs)]  
    while iterations < max_iterations:
        i_img = np.random.randint(0,n_imgs)
        n_data = pts2D_l[i_img].shape[0]
        if n_data == 0:
            continue
        test_idxs = np.random.randint(0,n_data,n_test)
        pts2D_test = pts2D_l[i_img][test_idxs,:]
        pts3D_test = pts3D_l[i_img][test_idxs,:]
        retval,rvecs_test,tvecs_test = cv2.solvePnP(pts3D_test,pts2D_test,K2,None,flags=cv2.SOLVEPNP_P3P)
        if not retval:
            continue
        T_m1_c_test = poses2matrix(tvecs_test,rvecs_test)
        T_m2_c = T_m2_c_l[i_img]
        T_c_m2 = T_inv(T_m2_c)
        T_m1_m2_test = np.dot(T_m1_c_test,T_c_m2)

        test_err = multiview_pnp_error(pts3D_l, pts2D_l, K2, T_m1_m2_test, T_m2_c_l)
        inlier_idxs = [np.where([e < max_reproj_error])[1].tolist() for e in test_err] # select indices of rows with accepted points

        if len_subelems(inlier_idxs) > len_subelems(best_inlier_idxs):
            T_m1_m2 = T_m1_m2_test
            best_inlier_idxs = inlier_idxs
            best_err = [np.array(e,dtype=np.int) for e in test_err]
            # print(T_m1_m2[:3,3])
        iterations+=1
        
    tvecs_l = [matrix2poses(T_m1_m2.dot(T_m2_c))[0] for T_m2_c in T_m2_c_l]
    rvecs_l = [matrix2poses(T_m1_m2.dot(T_m2_c))[1] for T_m2_c in T_m2_c_l]

    return T_m1_m2,tvecs_l,rvecs_l,best_inlier_idxs


def multiviewSolvePnPOptimization(pts3D_l, pts2D_l, poses_l, K2, T_m1_m2_init):

    if len(poses_l[0])>7:
        k0 = len(poses_l[0])-6
    else:
        k0 = 0
    T_m2_c_l = [pose2matrix(pose[k0:]) for pose in poses_l]

    x0 = matrix2pose(T_m1_m2_init)
    res=least_squares(optim_multiview_pnp_error,x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',args=(pts3D_l, pts2D_l, K2, T_m2_c_l))

    T_m1_m2 = pose2matrix(res.x)

    tvecs_l = [matrix2poses(T_m1_m2.dot(T_m2_c))[0] for T_m2_c in T_m2_c_l]
    rvecs_l = [matrix2poses(T_m1_m2.dot(T_m2_c))[1] for T_m2_c in T_m2_c_l]

    return T_m1_m2,tvecs_l,rvecs_l

def optim_multiview_pnp_error(x,pts3D_l, pts2D_l, K2, T_m2_c_l):
    T_m1_m2_test = np.eye(4)
    p0 = x[:3]
    q0 = x[3:]
    T_m1_m2_test[:3,3] = p0
    T_m1_m2_test[:3,:3] = Rotation.from_quat(q0).as_matrix()

    test_err = multiview_pnp_error(pts3D_l, pts2D_l, K2, T_m1_m2_test, T_m2_c_l)
    return np.sum([np.sum(e) for e in test_err]) / len_subelems(test_err)

def pnp_error(pts3D,pts2D,rvecs,tvecs,K2):
    pts2D_reproj = cv2.projectPoints(pts3D,rvecs,tvecs,K2,None)[0].reshape(-1,2)
    e = np.linalg.norm(pts2D-pts2D_reproj,axis=1)    
    return e

def multiview_pnp_error(pts3D_l, pts2D_l, K2, T_m1_m2_test, T_m2_c_l):
    n = len(pts2D_l)
    e_l = [None for _ in range(n)]
    for i in range(n):
        nfeatures = pts2D_l[i].shape[0]
        if nfeatures > 0:
            pts3D_i = pts3D_l[i]
            pts2D_i = pts2D_l[i]
            T_m1_ci = T_m1_m2_test.dot(T_m2_c_l[i])
            tvecs,rvecs = matrix2poses(T_m1_ci)
            e_l[i] = pnp_error(pts3D_i,pts2D_i,rvecs,tvecs,K2)
        else:
            e_l[i] = np.zeros(0)
    return e_l

def poses2matrix(tvecs,rvecs):
    R_ = cv2.Rodrigues(rvecs)[0]
    R = R_.T
    C = -R_.T.dot(tvecs)      
    T_m_c = np.eye(4)
    T_m_c[:3,:3] = R
    T_m_c[:3,3] = C.reshape(-1)
    return T_m_c

def pose2matrix(pose):
    p = pose[:3]
    q = pose[3:]
    R = Rotation.from_quat(q)
    T_m_c = np.eye(4)
    T_m_c[:3,:3] = R.as_matrix()
    T_m_c[:3,3] = p
    return T_m_c

def pq2matrix(pq):
    p = pq[0]
    q = pq[1]
    R = Rotation.from_quat(q)
    T_m_c = np.eye(4)
    T_m_c[:3,:3] = R.as_matrix()
    T_m_c[:3,3] = p
    return T_m_c

def matrix2poses(T_m_c):
    R = T_m_c[:3,:3]
    C = T_m_c[:3,3]

    rvecs = cv2.Rodrigues(R.T)[0]
    tvecs = -R.T.dot(C.reshape(3,1))      

    return tvecs,rvecs    

def matrix2pose(T_m_c):
    R = T_m_c[:3,:3]
    p = T_m_c[:3,3]
    q = Rotation.from_matrix(R).as_quat()

    pose = np.concatenate([p,q])

    return pose


def len_subelems(l):
    return np.sum([len(sl) for sl in l])    

def quat2matrix(q):
    Rot = Rotation.from_quat(q)
    R = Rot.as_matrix()
    return R

def T2rt(T_m_c1):
    R = T_m_c1[:3,:3].T
    tvec = (- R @ T_m_c1[:3,3]).reshape(-1)
    rvec = cv2.Rodrigues(R.T)[0]
    return tvec,rvec

def matrix2quat(R):
    Rot = Rotation.from_matrix(R)
    q = Rot.as_quat()
    return q
    
def fun_rectify_views(I_p,fov):

    equ = Equirectangular(I_p)    # Load equirectangular image
    
    phi = 0
    width = fov*equ._width/360
    height = fov*equ._height/180

    theta_1 = 0
    theta_2  = 180
    theta_3 = 90
    theta_4  = 270

    T1 = np.eye(4)
    T2 = np.eye(4)
    T3 = np.eye(4)
    T4 = np.eye(4)

    T1[:3,:3] = cv2.Rodrigues(theta_1*np.pi/180*np.array([0,1,0]))[0]
    T2[:3,:3] = cv2.Rodrigues(theta_2*np.pi/180*np.array([0,1,0]))[0]
    T3[:3,:3] = cv2.Rodrigues(theta_3*np.pi/180*np.array([0,1,0]))[0]
    T4[:3,:3] = cv2.Rodrigues(theta_4*np.pi/180*np.array([0,1,0]))[0]

    I1,K1 = equ.GetPerspective(fov, theta_1, phi, height, width)        
    I2,_  = equ.GetPerspective(fov, theta_2, phi, height, width)        
    I3,_  = equ.GetPerspective(fov, theta_3, phi, height, width)        
    I4,_  = equ.GetPerspective(fov, theta_4, phi, height, width)        

    I_list = [I1,I2,I3,I4]
    T_list = [T1,T2,T3,T4]

    return I_list, T_list, K1
