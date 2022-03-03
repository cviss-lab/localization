#!/usr/bin/env python3  
import rospy
import numpy as np
import sys
import os
from os.path import join, dirname, realpath
import tf
import tf2_ros
import tf2_msgs
from geometry_msgs.msg import Transform, PoseStamped, TransformStamped, Point, Quaternion
from std_msgs.msg import String
from nav_msgs.msg import Path
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from utils import *

hloc_module=join(dirname(realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import match_features, detect_features

class Node:

    def __init__(self):
        # self.test()
        rospy.init_node('self_localization')
        self.map_frame = rospy.get_param('~map_frame',default='map')
        self.debug = rospy.get_param('~debug',default=False)
        self.detector = rospy.get_param('~detector',default='SuperPoint')
        self.matcher = rospy.get_param('~matcher',default='SuperGlue')
        self.matcher_model = None
        self.detector_model = None
        self.load_models()

        self.results_dir = realpath('results')

        self.ls = tf.TransformListener()
        self.br = tf2_ros.StaticTransformBroadcaster()

        self.image = None        
        self.depth = None
        self.K = None
        self.timestamp = None
        self.timestamp_query = None
        self.robot_camera_frame_id = None
        self.query_camera_frame_id = None
        self.robot_pose = None
        self.unity_pose = None

        # set up robot image & depth subscriber
        sub1 = message_filters.Subscriber('image', Image)
        sub2 = message_filters.Subscriber('camera_info', CameraInfo)        
        sub3 = message_filters.Subscriber('depth', Image)   
        # set up trigger
        self.trigger_msg = None        
        # self.detector_running = False
        self.detector_running = True

        sub4 = rospy.Subscriber('trigger',String,self.callback_trigger)
        # set up query image subscriber
        sub5 = message_filters.Subscriber('image_query', Image)
        sub6 = message_filters.Subscriber('camera_info_query', CameraInfo)     

        ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2,sub3], 1, 0.5) 
        ts.registerCallback(self.callback)
   
        ts = message_filters.ApproximateTimeSynchronizer([sub5,sub6], 1, 0.5) 
        ts.registerCallback(self.callback_query)

        self.pub = rospy.Publisher('map_pose', PoseStamped, queue_size=1)

        transform = create_transform_stamped((0,0,0),
                                            (0,0,0,1),
                                            rospy.Time.now(),
                                            'unity',
                                            self.map_frame)
        self.br.sendTransform(transform)

        self.debug_img = None
        if self.debug:
            while True:
                if self.debug_img is not None:
                    cv2.imshow('debug',self.debug_img)
                    cv2.waitKey(3)


        rospy.spin()

    def callback_trigger(self,msg):
        self.trigger_msg = msg.data
        if self.detector_running:
            self.detector_running = False
        else:
            self.detector_running = True

    def callback(self, *args):
        try:
            self.timestamp = args[0].header.stamp
            self.robot_camera_frame_id = args[0].header.frame_id

            # self.ls.waitForTransform(self.map_frame, self.robot_camera_frame_id, rospy.Time(0), rospy.Duration(20.0))
            self.robot_pose = self.ls.lookupTransform(self.map_frame, self.robot_camera_frame_id, rospy.Time(0))

            cv_bridge = CvBridge()
            self.image = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
            self.K = np.array(args[1].K,dtype=np.float32).reshape(3,3)        
            self.depth = cv_bridge.imgmsg_to_cv2(args[2], desired_encoding='passthrough')

        except:
            return

    def callback_query(self,*args):
        print('query image recieved!')
        
        if not self.detector_running or self.image is None or self.robot_pose is None:
            return
        self.detector_running = False
            
        self.query_camera_frame_id = args[0].header.frame_id      
        self.timestamp_query = args[0].header.stamp            
        try:
            self.unity_pose = self.ls.lookupTransform(self.query_camera_frame_id, 'unity', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        cv_bridge = CvBridge()
        I2 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
        K2 = np.array(args[1].K,dtype=np.float32).reshape(3,3) 

        # I2 = args[0]
        # K2 = args[1]

        I1 = self.image
        D1 = self.depth
        K1 = self.K

        fname1 = join(self.results_dir,'kp1.h5')
        fname2 = join(self.results_dir,'kp2.h5')        

        kp1, des1 = self.feature_detection(I1, self.detector, fname1)
        kp2, des2 = self.feature_detection(I2, self.detector, fname2)
        
        matches = self.feature_matching(des1,des2,self.detector,self.matcher, fname1, fname2)

        # D1 = cv2.imread('D1.png',cv2.IMREAD_UNCHANGED)

        if len(matches) > 10:
            pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
            pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

            x_c,y_c,z_c = project_2d_to_3d(pts1.T,K1,D1)
            pts3D_c = np.array([x_c,y_c,z_c,np.ones(x_c.shape[0])])

            tc = self.robot_pose[0]
            qc = self.robot_pose[1]
            Tc = tf.transformations.quaternion_matrix(qc)
            Tc[:3,3] = tc

            pts3D = Tc.dot(pts3D_c)
            pts3D = pts3D[:3,:]/pts3D[3,:]

            pts3D = pts3D.T            
            
            idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
            pts3D = pts3D[idx]
            pts2D = pts2[idx]

            if len(idx) < 10:
                return
                 
            retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D, pts2D, K2, None,flags=cv2.SOLVEPNP_P3P)
            
            # find relocalized pose of query image relative to robot camera
            R_ = cv2.Rodrigues(rvecs)[0]
            R = R_.T
            C = -R_.T.dot(tvecs)  

            self.detector_running = False

            if self.debug:
                # for debugging purposes
                self.draw_matches(I1,I2,kp1,kp2,matches)
                
            # send localized pose relative to robot map
            self.send_reloc_pose(C,R)
            self.send_unity_map_pose()              

            self.detector_running = True  

    def send_reloc_pose(self,C,R):
        R2 = np.eye(4)
        R2[:3,:3] = R
        q = tf.transformations.quaternion_from_matrix(R2)     
        transform = create_transform_stamped((C[0],C[1],C[2]),
                                            (q[0],q[1],q[2],q[3]),
                                            self.timestamp,
                                            'reloc_pose',
                                            self.map_frame)
        self.br.sendTransform(transform)

    def send_unity_map_pose(self):
            
        (t_inv,q_inv) = self.unity_pose            

        transform = create_transform_stamped((t_inv[0],t_inv[1],t_inv[2]),
                              (q_inv[0],q_inv[1],q_inv[2],q_inv[3]),
                               self.timestamp_query,
                               'unity',
                               'reloc_pose')
        self.br.sendTransform(transform)

        # try:
        (t_unity,q_unity) = self.ls.lookupTransform('unity',self.map_frame , self.timestamp_query)
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # return 

        ps = PoseStamped()
        ps.header.stamp = self.timestamp
        ps.header.frame_id = 'unity'
        ps.pose.position.x = t_unity[0]
        ps.pose.position.y = t_unity[1]
        ps.pose.position.z = t_unity[2]
        ps.pose.orientation.x = q_unity[0]
        ps.pose.orientation.y = q_unity[1]
        ps.pose.orientation.z = q_unity[2]
        ps.pose.orientation.w = q_unity[3]

        self.pub.publish(ps)


    def draw_matches(self,I1,I2,kp1,kp2,matches):
        
        img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
        self.debug_img = img
        # plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()

    def feature_detection(self,I,detector,fname=None):
        
        if detector == 'SIFT':
            gray= cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
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

    def feature_matching(self,des1,des2,detector,matcher,fname1=None,fname2=None):

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

    def load_models(self):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            return     
        else:
            self.matcher_model = match_features.load_model(self.detector,self.matcher)
            self.detector_model = detect_features.load_model(self.detector)        

    def test(self):
        self.detector_running = True
        self.depth = cv2.imread('D1.png',cv2.IMREAD_UNCHANGED)
        self.image = cv2.imread('I1.jpg')
        self.K = np.loadtxt('K1.txt')
        K2 = np.loadtxt('K2.txt')
        I2 = cv2.imread('I2.jpg')
        self.callback_query(I2,K2)
        

if __name__ == '__main__':
    Node()    
    