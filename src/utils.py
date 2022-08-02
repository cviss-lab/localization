#!/usr/bin/env python3
import os
import shutil
import cv2
# import cv2.aruco as aruco
import numpy as np
# from geometry_msgs.msg import Transform, TransformStamped
from scipy.spatial.transform import Rotation
import json

def make_dir(path,delete_if_exists=False):   
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete_if_exists:
        shutil.rmtree(path)
        os.makedirs(path)


def pinhole_model(pose,K):
    """Loads projection and extrensics information for a pinhole camera model

    Arguments
    -------
        pose (numpy.array): px, py, cx, qx, qy, qz, qw
        K (numpy.array): camera matrix

    Returns
    -------
        P (numpy.array): projection matrix
        R (numpy.array): rotation matrix (camera coordinates)
        C (numpy.array): camera center (world coordinates)
    """

    pose = pose.reshape(-1)
    C = np.array(pose[1:4]).reshape(-1,1)

    # Convert camera rotation from quaternion to matrix
    q = pose[4:]
    Rot = Rotation.from_quat(q)
    
    # Find camera extrinsics (R,t)
    R = Rot.T
    t = Rot.T.dot(-C)

    # Construct projection matrix (P)
    P = np.dot(K, np.hstack([R, t]))

    return P,R,C  

def ProjectToImage(projectionMatrix, pos):
    """Project 3D world coordinates to 2D image coordinates using a pinhole camera model
    
    Arguments
    -------
        P (numpy.array): projection matrix
        pos (numpy.array): 3D world coordinates (3xN)

    Returns
    -------
        uv (numpy.array): 2D pixel coordinates (2xN)
    """    
    pos = np.array(pos).reshape(3,-1)
    pos_ = np.vstack([pos, np.ones((1, pos.shape[1]))])

    uv_ = np.dot(projectionMatrix, pos_)
    #uv_ = uv_[:, uv_[-1, :] > 0]
    uv = uv_[:-1, :]/uv_[-1, :]

    return uv

def ProjectToWorld(projectionMatrix, uv, R, C):
    """Back-project 2D image coordinates to rays in 3D world coordinates using a pinhole camera model
    
    Arguments
    -------
        P (numpy.array): projection matrix
        uv (numpy.array): 2D pixel coordinates (2xN)
        R (numpy.array): rotation matrix (camera coordinates)
        C (numpy.array): camera center (world coordinates)       

    Returns
    -------
        pos (numpy.array): [3D world coordinates (3xN)]
    """       
    uv_ = np.vstack([uv[0,:], uv[1,:], np.ones((1, uv.shape[1]))])
    pinvProjectionMatrix = np.linalg.pinv(projectionMatrix)

    pos2_ = np.dot(pinvProjectionMatrix, uv_)
    pos2_[-1,pos2_[-1,:]==0] = 1
    pos2 = pos2_[:-1,:]/pos2_[-1,:]
    rays = pos2-C

    # check that rays project forwards
    rays_local = np.dot(R , rays)
    rays[:,rays_local[2,:]<0] = -1*rays[:,rays_local[2,:]<0]    
    rays = rays/np.linalg.norm(rays,axis=0)

    return rays


def Tmatrix_inverse(Tmat):
    if Tmat.shape == (4,4):
        R = Tmat[:3,:3]
        t = Tmat[:3,3]
        R_inv = R.T
        t_inv = -R_inv.dot(t)     
        Tmat_inv = np.eye(4)
        Tmat_inv[:3,:3] = R_inv
        Tmat_inv[:3,3] = t_inv
        return Tmat_inv
    else:
        raise 'Error: input should be 4x4 transformation matrix'

def create_transform(p,q):
    
    T = Transform()
    T.translation.x = p[0]
    T.translation.y = p[1]
    T.translation.z = p[2]
    
    T.rotation.x = q[0]
    T.rotation.y = q[1]
    T.rotation.z = q[2]
    T.rotation.w = q[3]
    
    return T

def create_transform_stamped(p,q,t,child_frame_id,frame_id):
    
    T = TransformStamped()
    T.header.stamp = t
    T.header.frame_id = frame_id
    T.child_frame_id = child_frame_id

    T.transform.translation.x = p[0]
    T.transform.translation.y = p[1]
    T.transform.translation.z = p[2]

    T.transform.rotation.x = q[0]
    T.transform.rotation.y = q[1]
    T.transform.rotation.z = q[2]
    T.transform.rotation.w = q[3]
    
    return T

def unpack_transform(T):
    p = [T.translation.x, T.translation.y, T.translation.z]
    q = [T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w]
    return p,q

def unpack_pose(T):
    p = [T.position.x, T.position.y, T.position.z]
    q = [T.orientation.x, T.orientation.y, T.orientation.z, T.orientation.w]
    return p,q    

def points2numpy(pl):
    return np.array([[p.x,p.y,p.z] for p in pl])

def quaterions2numpy(ql):
    return np.array([[q.x,q.y,q.z,q.w] for q in ql])    


def project_2d_to_3d(m,K,D,center=False, h=0):
    u = m[0,:]
    v = m[1,:]
         
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    d = []

    if center:
        d0 = D[int(v.mean()),int(u.mean())]   
        d = [d0 for _ in range(u.shape[0])]  
    else:
        for ui,vi in zip(u,v):
            di = D[int(vi)-h:int(vi)+h+1,int(ui)-h:int(ui)+h+1]
            if len(di)>0:
                di = di[di>0].mean()
                d.append(di)

    Z = np.array(d,dtype=np.float)/1000
    X = Z*(u-cx)/fx
    Y = Z*(v-cy)/fy    

    return X,Y,Z      

def countour2mask(m,I):
    
    mask = np.zeros(I.shape)
    contours=m
    mask=cv2.drawContours(mask, contours, -1, (255),1)    
    return mask

def cloud_inside_polygon(m,K,D):
    mask = countour2mask(m,D)
    cloud = project_2d_to_3d(mask,K,D)
    return cloud

def detect_markers(I,find_ids=None,xy_array=False,ignore_zeros=True):

    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    w = int(I.shape[1])
    h = int(I.shape[0])

    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        # "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        # "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        # "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        # "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        # "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        # "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        # "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        # "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        # "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        # "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        # "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        # "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        # "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        # "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        # "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        # "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        # "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        # "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        # "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    # loop over the types of ArUco dictionaries
    for (arucoName, arucoDict) in ARUCO_DICT.items():
        
        # load the ArUCo dictionary, grab the ArUCo parameters, and
        # attempt to detect the markers for the current dictionary
        arucoDict = cv2.aruco.Dictionary_get(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        arucoParams.adaptiveThreshConstant = 15
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            gray, arucoDict, parameters=arucoParams)    

        if ids is not None:
            # print(arucoName)
            break

    if ids is None:
        return []
    ids = list(ids.reshape(-1))

    # if specified, only find markers with ids in find_ids list
    if isinstance(find_ids,list):
        corners2 = []
        ids2 = []
        for i,c in zip(ids,corners):
            if i in find_ids:
                ids2.append(i)
                corners2.append(c)
    else:
        corners2 = corners
        ids2 = ids

    objects = []
    for c,i in zip(corners2,ids2):
        con = c.reshape(-1, 2)
        if con.shape[0] < 3:
            continue

        if ignore_zeros and ids==0:
            continue

        obj = dict([])
        obj['id'] = 'ID:'+str(i)
        obj['confidence'] = 1.0

        if xy_array:
            coords = np.array([[],[]])
            for pt in con:
                x = float(pt[0])
                y = float(pt[1])
                coords = np.hstack([coords,[[x],[y]]])
            obj['coords'] = coords
        else:
            obj['coords'] = []
            for pt in con:
                coords = dict([])
                coords['x'] = float(pt[0])
                coords['y'] = float(pt[1])
                obj['coords'].append(coords)

        objects.append(obj)     

    return objects


def detect_manual(I,m,s,h=0.05):

    W = int(I.shape[1])
    H = int(I.shape[0])

    objects = []

    obj = dict([])
    obj['id'] = 'ID:'+str(s)
    obj['confidence'] = 1.0

    coords = np.array([[],[]])
    for i in range(4):
        if i == 2:
            h1 = 1
            h2 = -1
        elif i == 1:
            h1 = 1
            h2 = 1
        elif i == 0:
            h1 = -1
            h2 = 1
        elif i == 3:
            h1 = -1
            h2 = -1                                    
        x = (float(m[1])+h1*h)*W
        y = (float(m[0])+h2*h)*H
        coords = np.hstack([coords,[[x],[y]]])
    obj['coords'] = coords


    objects.append(obj)     

    return objects

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def return_T_M2_C2(sfm_file):
    # sfm_file = '/home/jp/Desktop/Rishabh/Handheld/1m_debug6/0_6_less_img_ultra/reconstruction_global/sfm_data.json'
    with open(sfm_file, 'r') as f:
        sfm = json.load(f)

    T_m2_c2_dict = {}
    for i in range(len(sfm['views'])):
        # print(sfm['views'])
        rotation = np.array(sfm['extrinsics'][i]['value']['rotation'])
        centers = np.array(sfm['extrinsics'][i]['value']['center'])
        query_idx = int(sfm['views'][i]['value']['ptr_wrapper']['data']['filename'].replace('.jpg', ''))
        T_m2_c2 = np.eye(4)
        R = rotation.T
        # t = R.dot(-centers).reshape(-1, 1)
        C = centers.reshape((-1, 1))
        # Tm2_c2 = np.hstack([R, t])
        T_m2_c2[:3, :3] = R
        T_m2_c2[:3, 3] = C.reshape(-1)
        T_m2_c2_dict[query_idx] = T_m2_c2
        # T_m2_c2_list.append(T_m2_c2)
    return T_m2_c2_dict
