#!/usr/bin/env python3  
import rospy
import numpy as np
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
# from matplotlib import pyplot as plt

class Node:

    def __init__(self):
        # self.test()
        rospy.init_node('self_localization')
        self.map_id = rospy.get_param('~map_id',default='map')
        self.save = rospy.get_param('~save',default=False)
        self.debug = rospy.get_param('~debug',default=False)

        self.ls = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        self.image = None        
        self.depth = None
        self.K = None
        self.timestamp = None
        self.timestamp_query = None
        self.robot_camera_frame_id = None
        self.query_camera_frame_id = None

        # set up robot image & depth subscriber
        sub1 = message_filters.Subscriber('image', Image)
        sub2 = message_filters.Subscriber('camera_info', CameraInfo)        
        sub3 = message_filters.Subscriber('depth', Image)   

        # set up trigger
        self.trigger_msg = None        
        self.detector_running = False
        rospy.Subscriber('trigger',String,self.callback_trigger)

        ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2,sub3], 1, 0.5) 
        ts.registerCallback(self.callback)

        # set up query image subscriber
        sub1 = message_filters.Subscriber('image_query', Image)
        sub2 = message_filters.Subscriber('camera_info_query', CameraInfo)        
        # sub3 = message_filters.Subscriber('pose_query', PoseStamped)        

        ts = message_filters.ApproximateTimeSynchronizer([sub1,sub2], 1, 0.5) 
        ts.registerCallback(self.callback_query)

        self.pub = rospy.Publisher('map_pose', PoseStamped, queue_size=1)

        rospy.spin()

    def callback_trigger(self,msg):
        self.trigger_msg = msg.data
        if self.detector_running:
            self.detector_running = False
        else:
            self.detector_running = True

    def callback(self, *args):

        if not self.detector_running:
            return

        self.timestamp = args[0].header.stamp
        self.robot_camera_frame_id = args[0].header.frame_id
        cv_bridge = CvBridge()
        self.image = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
        self.K = np.array(args[1].K,dtype=np.float32).reshape(3,3)        
        self.depth = cv_bridge.imgmsg_to_cv2(args[2], desired_encoding='passthrough')

    def callback_query(self,*args):
        
        if not self.detector_running or self.image is None:
            return
        cv_bridge = CvBridge()
        I2 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
        K2 = np.array(args[1].K,dtype=np.float32).reshape(3,3) 
        self.query_camera_frame_id = args[0].header.frame_id      
        self.timestamp_query = args[0].header.stamp
        
        if self.save:                 
            cv2.imwrite('results/%i_query.jpg' % self.timestamp_query.secs,I2)
            cv2.imwrite('results/%i_image.jpg' % self.timestamp_query.secs, self.image)
            cv2.imwrite('results/%i_depth.png' % self.timestamp_query.secs, self.depth)            

        # I2 = args[0]
        # K2 = args[1]

        I1 = self.image
        D1 = self.depth
        K1 = self.K

        kp1, des1 = self.feature_detection(I1)
        kp2, des2 = self.feature_detection(I2)

        matches = self.feature_matching(des1,des2)

        # D1 = cv2.imread('D1.png',cv2.IMREAD_UNCHANGED)

        if len(matches) > 10:
            pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
            pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

            x,y,z = project_2d_to_3d(pts1.T,K1,D1)
            pts3D = np.array([x,y,z]).T            
            
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

            if self.save:
                P = np.dot(K2, np.hstack([R_, tvecs]))                
                np.savetxt('results/%i_P.txt'  % self.timestamp_query.secs,P)
                np.savetxt('results/%i_K1.txt' % self.timestamp_query.secs,K1)
                np.savetxt('results/%i_K2.txt' % self.timestamp_query.secs,K2)

            if self.debug:
                # for debugging purposes
                self.draw_matches(I1,I2,kp1,kp2,matches)
                
            # send localized pose relative to robot map
            self.send_reloc_pose(C,R)
            self.send_unity_map_pose()                

    def send_reloc_pose(self,C,R):
        ps = PoseStamped()
        R2 = np.eye(4)
        R2[:3,:3] = R
        q = tf.transformations.quaternion_from_matrix(R2)
        ps.header.stamp = self.timestamp
        ps.header.frame_id = self.robot_camera_frame_id
        ps.pose.position.x = C[0]
        ps.pose.position.y = C[1]
        ps.pose.position.z = C[2]
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        self.ls.waitForTransform(self.map_id, self.robot_camera_frame_id, self.timestamp, rospy.Duration(15.0))
        try:
            ps2 = self.ls.transformPose(self.map_id,ps)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return      
        self.br.sendTransform((ps2.pose.position.x,ps2.pose.position.y,ps2.pose.position.z),
                              (ps2.pose.orientation.x,ps2.pose.orientation.y,ps2.pose.orientation.z,ps2.pose.orientation.w),
                              self.timestamp,
                              'reloc_pose',
                              self.map_id)

    def send_unity_map_pose(self):

        try:
            (t_inv,q_inv) = self.ls.lookupTransform(self.query_camera_frame_id, self.map_id, self.timestamp_query)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
            
        self.br.sendTransform((t_inv[0],t_inv[1],t_inv[2]),
                              (q_inv[0],q_inv[1],q_inv[2],q_inv[3]),
                               self.timestamp,
                               'unity',
                               'reloc_pose')

        try:
            (t_unity,q_unity) = self.ls.lookupTransform('unity',self.map_id , self.timestamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

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
        # plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()
        if self.save:
            cv2.imwrite('results/%i_matches.jpg' % self.timestamp_query.secs,img)          

    def feature_detection(self,I):
        # find the keypoints and descriptors with ORB
        # orb = cv2.ORB_create()
        # kp, des = orb.detectAndCompute(I,None)

        gray= cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
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
        
        # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # top_N = len(des1)
        # matches = matches[:top_N]
        
        # # Draw first 10 matches.
        # img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # img3 = cv2.drawMatchesKnn(I1,kp1,I2,kp2,good,None,flags=2)
        # plt.imshow(img3),plt.show()   
        return matches     

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
    