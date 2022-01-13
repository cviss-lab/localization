#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from utils import *
import os
import sys
import json

hloc_module=os.path.join(os.path.dirname(os.path.realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import search_images

class Node:

    def __init__(self):
        rospy.init_node('image_search')

        self.num_matches = rospy.get_param('~num_matches',default=10)

        self.running = False
        self.dataset = os.path.join(hloc_module,'datasets')
        self.outputs = os.path.join(hloc_module,'outputs')
        self.query_dir = os.path.join(self.dataset,'query')
        self.projection_matrices = self.init_projection_matrices()
        self.K = None
        
        # self.debug3d()

        rospy.Subscriber('image',Image,self.callback)
        rospy.Subscriber('trigger',String,self.callback_trigger)
        rospy.Subscriber('trigger_debug',String,self.callback_trigger_debug)
      
        rospy.spin()

    def init_projection_matrices(self):

        poses_file = os.path.join(self.dataset,'poses.csv')
        intrinsics_file = os.path.join(self.dataset,'intrinsics.json')

        with open(intrinsics_file,'r') as f:
            self.intrinsics = json.load(f)
            self.K = np.array(self.intrinsics['camera_matrix'])

        poses = np.loadtxt(poses_file, delimiter=",")

        projection_matrices = []
        for p in poses:
            projection_matrices.append(pinhole_model(p,self.K)[0])

        return projection_matrices


    def callback_trigger(self,msg):
        self.trigger_msg = msg.data
        if self.running:
            self.running = False
        else:
            self.running = True

    def callback_trigger_debug(self,msg):
        self.running = True            

    def callback(self, *args):

        if not self.running:
            return

        cv_bridge = CvBridge()
        query_image = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough') 
        query_id = str(args[0].header.seq)
        f_query = os.path.join(self.query_dir,query_id+'.jpg')
        cv2.imwrite(f_query,query_image)

        search_images.main(self.dataset,self.outputs,self.num_matches,False)  
        
        image_files = os.listdir(os.path.join(self.outputs,'matches',query_id))
        image_ids = [int(f[0].split('.')[0]) for f in image_files]
        image_files = [os.path.join(self.outputs,'matches',query_id,f) for f in image_files]
        
        proj_matrices = [self.projection_matrices[i] for i in image_ids]

        self.triangulate_matches(image_files,query_image,proj_matrices)

    def triangulate_matches(self,image_files,query_image,proj_matrices):
        features_list = []
        desc_list = []
        kp_q, des_q = self.feature_detection(query_image)
        
        matches_mat = np.empty((len(kp_q),len(image_files))); matches_mat[:] = np.nan
        dist_mat    = np.empty((len(kp_q),len(image_files))); dist_mat[:] = np.nan

        for image_file in image_files:
            img = cv2.imread(image_file)
            kp, des = self.feature_detection(img)
            features_list.append(kp)
            desc_list.append(des)

        for i,des in enumerate(desc_list):
            matches = self.feature_matching(des_q,des)
            matches_mat[[m.queryIdx for m in matches],i] = [m.trainIdx for m in matches]
            dist_mat[[m.queryIdx for m in matches],i] = [m.distance for m in matches]

        # query_2D = np.float32([kp.pt for kp in kp_q])
        query_3D = []
        query_2D = []
        for k, (matches, dist) in enumerate(zip(matches_mat, dist_mat)):
            if np.sum(dist>0) < 3:
                continue
            ind = np.argsort(dist)

            i1 = ind[0]
            i2 = ind[1]   
            i3 = ind[2]         

            m1 = int(matches[i1])
            m2 = int(matches[i2])
            m3 = int(matches[i3])

            p1 = np.float32(features_list[i1][m1].pt).reshape(2,1)
            p2 = np.float32(features_list[i2][m2].pt).reshape(2,1)
            p3 = np.float32(features_list[i3][m3].pt).reshape(2,1)

            P1 = proj_matrices[i1]
            P2 = proj_matrices[i2]
            P3 = proj_matrices[i3]

            pt_3d = cv2.triangulatePoints(P1,P2,p1,p2)
            pt_3d = (pt_3d[:3]/pt_3d[3]).reshape(-1)
            
            p3_proj = ProjectToImage(P3, pt_3d)
            
            query_3D.append(pt_3d)
            query_2D.append(np.float32(kp_q[k].pt))
        
        query_2D = np.array(query_2D)
        query_3D = np.array(query_3D)

        np.savetxt(os.path.join(self.outputs,'query_3D.txt'),query_3D)
        # self.debug3d()
        # retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(query_3D, query_2D, self.K, None,flags=cv2.SOLVEPNP_P3P)

    def feature_detection(self,I):
        # find the keypoints and descriptors with ORB
        # orb = cv2.ORB_create()
        # kp, des = orb.detectAndCompute(I,None)

        if I.ndim==3:
            gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        else:
            gray = I
        
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)

        return kp, des

    def feature_matching(self,des1,des2):
        # match the descriptors
        # create BFMatcher object

        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1,des2, k=2)

        # # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good
          
        return matches     
        
    def debug3d(self):
        import matplotlib.pyplot as plt
        import numpy as np
        query_3D = np.loadtxt(os.path.join(self.outputs,'query_3D.txt'))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(query_3D[:,0],query_3D[:,1],query_3D[:,2])
        plt.show()

if __name__ == '__main__':
    Node()    
    