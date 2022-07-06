from __future__ import print_function

import urllib.request
import bz2
import os
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer

class BundleAdjustment():
    def __init__(self,camera_params=None,n_cameras=None, n_points=None, camera_indices=None, point_indices=None, points_2d=None, points_3d=None):
        self.camera_params = camera_params
        self.n_cameras = n_cameras
        self.n_points = n_points 
        self.camera_indices = camera_indices 
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.points_3d = points_3d

        self.A = None

    def read_bal_data(self,file_name):
        with bz2.open(file_name, "rt") as file:
            n_cameras, n_points, n_observations = map(
                int, file.readline().split())

            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2))

            for i in range(n_observations):
                camera_index, point_index, x, y = file.readline().split()
                camera_indices[i] = int(camera_index)
                point_indices[i] = int(point_index)
                points_2d[i] = [float(x), float(y)]

            camera_params = np.empty(n_cameras * 9)
            for i in range(n_cameras * 9):
                camera_params[i] = float(file.readline())
            camera_params = camera_params.reshape((n_cameras, -1))

            points_3d = np.empty(n_points * 3)
            for i in range(n_points * 3):
                points_3d[i] = float(file.readline())
            points_3d = points_3d.reshape((n_points, -1))

        self.camera_params = camera_params
        self.n_cameras = n_cameras
        self.n_points = n_points 
        self.camera_indices = camera_indices 
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.points_3d = points_3d

    def bundle_adjustment_sparsity(self):
        m = self.camera_indices.size * 2
        n = self.n_cameras * 9 + self.n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.camera_indices.size)
        for s in range(9):
            A[2 * i, self.camera_indices * 9 + s] = 1
            A[2 * i + 1, self.camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, self.n_cameras * 9 + self.point_indices * 3 + s] = 1
            A[2 * i + 1, self.n_cameras * 9 + self.point_indices * 3 + s] = 1

        self.A = A

    def rotate(self,points, rot_vecs):
        """Rotate points by given rotation vectors.
        
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(self,points,camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        n = np.sum(points_proj**2, axis=1)
        r = 1 + k1 * n + k2 * n**2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj

    def fun(self,params):
        """Compute residuals.
        
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:self.n_cameras * 9].reshape((self.n_cameras, 9))
        points_3d = params[self.n_cameras * 9:].reshape((self.n_points, 3))
        points_proj = self.project(points_3d[self.point_indices], camera_params[self.camera_indices])
        return (points_proj - self.points_2d).ravel()                    

    def fun_init(self):
        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        return self.fun(x0)        

    def least_squares(self):
        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        res = least_squares(self.fun, x0, jac_sparsity=self.A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
        return res

class BundleAdjustment2():
    def __init__(self,camera_params=None,n_cameras=None, n_points=None, camera_indices=None, point_indices=None, 
                 points_2d=None, points_3d=None, intrinsics=None,optimize_points=True,optimize_tvec=True,optimize_rvec=True,
                 attach_cameras=False):
        self.camera_params = camera_params
        self.n_cameras = n_cameras
        self.n_points = n_points 
        self.camera_indices = camera_indices 
        self.point_indices = point_indices
        self.points_2d = points_2d
        self.points_3d = points_3d
        self.intrinsics = intrinsics
        self.tvecs_init=[]
        self.rvecs_init=[]
        
        self.A = None
        self.k_cam_params = 6

        self.optimize_points = optimize_points
        self.optimize_tvec = optimize_tvec
        self.optimize_rvec = optimize_rvec
        self.attach_cameras = attach_cameras

        # if optimize_tvec and optimize_rvec:
        #     self.k_cam_params = 6
        # elif optimize_tvec or optimize_rvec:
        #     self.k_cam_params = 3
        # else:
        #     self.k_cam_params = 0


    def init_viz(self):
        xp,yp,zp = np.mean(self.points_3d,axis=0)
        self.visualizer = CameraPoseVisualizer([xp-5, xp+5], [yp-5, yp+5], [zp-2, zp+8])

    def add_viz(self,color):

        for i,pose in enumerate(self.camera_params):
            if self.optimize_rvec:
                rvecs = pose[3:]
            else:
                rvecs = self.rvecs_init[i]                 
                            
            if self.optimize_tvec:
                if self.attach_cameras:
                    pose0 = self.camera_params[0]
                    tvecs0 = pose0[:3]
                    rvecs0 = pose0[3:]
                    Rot0 = cv2.Rodrigues(rvecs0)[0] 
                    C0 = -Rot0.T.dot(tvecs0) 
                    Rot = cv2.Rodrigues(rvecs)[0] 
                    tvecs = -Rot.dot(C0) 
                else:
                    tvecs = pose[:3]            
            else:
                tvecs = self.tvecs_init[i].reshape(-1)
                Rot_init = cv2.Rodrigues(self.rvecs_init[i])[0] 
                C = -Rot_init.T.dot(tvecs) 
                Rot = cv2.Rodrigues(rvecs)[0] 
                tvecs = -Rot.dot(C) 

            Rot = cv2.Rodrigues(rvecs)[0] 
            T = np.eye(4)
            T[:3,:3] = Rot.T            
            T[:3,3] = -Rot.T.dot(tvecs)

            self.visualizer.extrinsic2pyramid(T, color, 1)
            self.visualizer.plot_points(self.points_3d,color)

    def read_from_data(self,pts3D,pts_idx,pts2D_l,tvecs_l,rvecs_l,K,dist=[0,0,0,0]):
        self.points_2d = np.vstack(pts2D_l)
        self.points_3d = pts3D        
        self.point_indices = np.hstack(pts_idx)
        self.camera_indices = np.hstack([i*np.ones(len(pts2D)) for i,pts2D in enumerate(pts2D_l)])
        self.camera_indices = np.array(self.camera_indices,dtype=np.int)

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        k1,k2,p1,p2 = dist

        self.intrinsics = fx,fy,cx,cy,k1,k2,p1,p2
        self.n_points = self.points_3d.shape[0] 
        self.n_cameras = len(tvecs_l)

        self.camera_params = np.zeros((self.n_cameras,self.k_cam_params))
        self.camera_params[:,:3] = np.array(tvecs_l,dtype=np.float32).reshape(-1,3)        
        self.camera_params[:,3:] = np.array(rvecs_l,dtype=np.float32).reshape(-1,3)
        self.tvecs_init = tvecs_l
        self.rvecs_init = rvecs_l

        self.tvecs = self.tvecs_init
        self.rvecs = self.rvecs_init

        # if self.optimize_tvec:
        #     self.camera_params[:,:3] = np.array(tvecs_l,dtype=np.float32).reshape(-1,3)
        # if self.optimize_rvec:
        #     self.camera_params[:,3:] = np.array(rvecs_l,dtype=np.float32).reshape(-1,3)
        
    def bundle_adjustment_sparsity(self):
        k = self.k_cam_params
        m = self.camera_indices.size * 2
        if self.optimize_points:
            n = self.n_cameras * k + self.n_points * 3
        else:
            n = self.n_cameras * k
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(self.camera_indices.size)
        # if self.optimize_tvec and self.optimize_rvec:            
        #     range_k = range(k)
        # elif self.optimize_tvec:            
        #     range_k = range(0,3)
        # elif self.optimize_rvec:            
        #     range_k = range(3,6)
        # else:            
        #     range_k = range(0)      

        for s in range(k):
            A[2 * i, self.camera_indices * k + s] = 1
            A[2 * i + 1, self.camera_indices * k + s] = 1

        if self.optimize_points:
            for s in range(3):
                A[2 * i, self.n_cameras * k + self.point_indices * 3 + s] = 1
                A[2 * i + 1, self.n_cameras * k + self.point_indices * 3 + s] = 1

        self.A = A

    def project(self,points,poses):
        """Convert 3-D points to 2-D by projecting onto images."""

        fx,fy,cx,cy,k1,k2,p1,p2 = self.intrinsics        
        K = np.array([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,  0,   1]])
        dist = np.array([k1,k2,p1,p2],dtype=np.float32)
        points_proj = np.zeros((len(self.point_indices),2))
        for i,pose in enumerate(poses):
            
            if self.optimize_rvec:
                rvecs = pose[3:]
            else:
                rvecs = self.rvecs_init[i]                 

            if self.optimize_tvec:
                if self.attach_cameras:
                    pose0 = self.camera_params[0]
                    tvecs0 = pose0[:3]
                    rvecs0 = pose0[3:]
                    Rot0 = cv2.Rodrigues(rvecs0)[0] 
                    C0 = -Rot0.T.dot(tvecs0) 
                    Rot = cv2.Rodrigues(rvecs)[0] 
                    tvecs = -Rot.dot(C0) 
                else:
                    tvecs = pose[:3]            
            else:
                tvecs = self.tvecs_init[i].reshape(-1)
                Rot_init = cv2.Rodrigues(self.rvecs_init[i])[0] 
                C = -Rot_init.T.dot(tvecs) 
                Rot = cv2.Rodrigues(rvecs)[0] 
                tvecs = -Rot.dot(C)                                 

            cam_i = self.camera_indices == i
            points_proj[cam_i,:] = cv2.projectPoints(points[cam_i,:],rvecs,tvecs,K,dist)[0].reshape(-1,2)
        
        return points_proj

    def save_poses(self,params):

        k = self.k_cam_params
        camera_params = params[:self.n_cameras * k].reshape((self.n_cameras, k))

        for i,pose in enumerate(camera_params):
            
            if self.optimize_rvec:
                rvecs = pose[3:]
            else:
                rvecs = self.rvecs_init[i]                 

            if self.optimize_tvec:
                if self.attach_cameras:
                    pose0 = self.camera_params[0]
                    tvecs0 = pose0[:3]
                    rvecs0 = pose0[3:]
                    Rot0 = cv2.Rodrigues(rvecs0)[0] 
                    C0 = -Rot0.T.dot(tvecs0) 
                    Rot = cv2.Rodrigues(rvecs)[0] 
                    tvecs = -Rot.dot(C0) 
                else:
                    tvecs = pose[:3]            
            else:
                tvecs = self.tvecs_init[i].reshape(-1)
                Rot_init = cv2.Rodrigues(self.rvecs_init[i])[0] 
                C = -Rot_init.T.dot(tvecs) 
                Rot = cv2.Rodrigues(rvecs)[0] 
                tvecs = -Rot.dot(C)                                 

            self.rvecs[i] = rvecs
            self.tvecs[i] = tvecs         

    def fun(self,params):
        """Compute residuals.
        
        `params` contains camera parameters and 3-D coordinates.
        """
        k = self.k_cam_params
        camera_params = params[:self.n_cameras * k].reshape((self.n_cameras, k))
        if self.optimize_points:
            points_3d = params[self.n_cameras * k:].reshape((self.n_points, 3))
        else:
            points_3d = self.points_3d
        points_proj = self.project(points_3d[self.point_indices], camera_params)
        return (points_proj - self.points_2d).ravel()               

    def fun_init(self):
        if self.optimize_points:
            x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        else:
            x0 = self.camera_params.ravel()
        return x0, self.fun(x0)        

    def least_squares(self):
        x0,_ = self.fun_init()
        res = least_squares(self.fun, x0, jac_sparsity=self.A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
        params = res.x
        self.camera_params = params[:self.n_cameras * self.k_cam_params].reshape((self.n_cameras, self.k_cam_params))

        if self.optimize_points:
            self.points_3d = params[self.n_cameras * self.k_cam_params:].reshape((self.n_points, 3))

        self.save_poses(params)

        return res

def main():

    BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
    FILE_NAME = "problem-49-7776-pre.txt.bz2"
    URL = BASE_URL + FILE_NAME

    if not os.path.isfile(FILE_NAME):
        urllib.request.urlretrieve(URL, FILE_NAME)

    bundle_adjustment = BundleAdjustment()

    bundle_adjustment.read_bal_data(FILE_NAME)

    n_cameras = bundle_adjustment.camera_params.shape[0]
    n_points = bundle_adjustment.points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * bundle_adjustment.points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    _,f0 = bundle_adjustment.fun_init()
    plt.plot(f0)

    bundle_adjustment.bundle_adjustment_sparsity()
    t0 = time.time()

    res = bundle_adjustment.least_squares()
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    plt.plot(res.fun)

    plt.show()

if __name__ == '__main__':
    main()