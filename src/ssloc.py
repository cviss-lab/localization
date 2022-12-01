#!/usr/bin/env python3  
import scipy
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
import utils
import matplotlib.pyplot as plt
from ssloc_offline import ssloc

class Node:

    def __init__(self):
        
        self.timestamp = None
        self.timestamp_query = None
        self.query_camera_frame_id = None
        self.tf_timout = 1.5
        self.counter = 0
        
        self.K1 = None
        self.currently_running = False
        self.frames_odom = dict()
        self.requests_timeout = dict()
        self.prev_request = dict()
        self.base_link = dict()
        self.N_msloc = dict()
        self.query_images = dict()
        self.query_poses = dict()
        self.retrieved_anchors = dict()

        # initialize parameters
        rospy.init_node('self_localization')
        self.map_frame_id = rospy.get_param('~map_frame_id',default='map')
        self.robot_camera_frame_id = rospy.get_param('~camera_frame_id',default=None)
        self.send_unity_pose = rospy.get_param('~send_unity_pose',default=False)        
        self.frame_rate = rospy.get_param('~frame_rate',default=0.5)        
        self.detector = rospy.get_param('~detector',default='loftr')
        self.matcher = rospy.get_param('~matcher',default='loftr')  
        self.sliding_average_buffer = rospy.get_param('~sliding_average_buffer',default=1)       
        self.num_retrieved_anchors = rospy.get_param('~num_retrieved_anchors',default=5)        

        self.results_dir = rospy.get_param('~save_directory',default='results')
        self.create_new_anchors = rospy.get_param('~create_new_anchors',default=False)        
        self.num_query_devices = rospy.get_param('/num_users',default=1) 

        # load models
        self.ssloc = ssloc(results_dir=self.results_dir, create_new_anchors=self.create_new_anchors, detector=self.detector, matcher=self.matcher)

        utils.make_dir(join(self.results_dir), delete_if_exists=self.create_new_anchors)
        utils.make_dir(join(self.results_dir,'local_features'))
        utils.make_dir(join(self.results_dir,'global_features'))
        utils.make_dir(join(self.results_dir,'rgb'))
        utils.make_dir(join(self.results_dir,'depth'))
        utils.make_dir(join(self.results_dir,'poses'))

        # register publishers & subscribers then spin node

        self.ls = tf.TransformListener()
        self.br = tf2_ros.StaticTransformBroadcaster()

        # set up robot image & depth subscriber
        sub1 = message_filters.Subscriber('image', Image)
        sub3 = message_filters.Subscriber('depth', Image)   
        rospy.Subscriber('camera_info',CameraInfo, self.camera_info_callback)

        if self.create_new_anchors:
            ts = message_filters.ApproximateTimeSynchronizer([sub1,sub3], 1, 0.5) 
            ts.registerCallback(self.create_anchor)

        print('subscribed to {} query devices!'.format(self.num_query_devices))

        # add subscribers for players
        for i in range(self.num_query_devices):
            image_query = "/Player{}/camera/image/compressed".format(str(i))
            camera_info_query = "/Player{}/camera/camera_info".format(str(i))
            pose_query = "/Player{}/camera/pose".format(str(i))
            query_frame_id = 'Player{}'.format(str(i))
            query_odom_frame = 'Player{}_unity'.format(str(i))
            self.add_player_subs(image_query,camera_info_query,pose_query,query_frame_id,
                                 query_odom_frame,requests_timeout=0,base_link=False,N_msloc=0, retrieved_anchors=5)

        # add subscribers for tello drone
        image_query = "/tello/camera/image_raw/compressed"
        camera_info_query = "/tello/camera/camera_info"
        pose_query = "/tello/real_world_pos"
        query_frame_id = 'tello_camera'
        query_odom_frame = 'tello_camera_odom'
        self.add_player_subs(image_query,camera_info_query,pose_query,query_frame_id,
                             query_odom_frame,requests_timeout=30,base_link=True,N_msloc=0, retrieved_anchors=5)

        self.pub = rospy.Publisher('reloc_map_pose', PoseStamped, queue_size=1)
        self.pub2 = rospy.Publisher('retrieved_image', Image, queue_size=1)
        self.pub3 = rospy.Publisher('matches_image', Image, queue_size=1)

        rospy.spin()

    def add_player_subs(self,image_query,camera_info_query,pose_query,query_frame_id,
                        query_odom_frame,requests_timeout=0,base_link=False,N_msloc=0, retrieved_anchors=5):
                        
        sub5 = message_filters.Subscriber(image_query, CompressedImage)
        sub6 = message_filters.Subscriber(camera_info_query, CameraInfo)     
        sub7 = message_filters.Subscriber(pose_query, PoseStamped)     

        if self.send_unity_pose:
            ts = message_filters.ApproximateTimeSynchronizer([sub5,sub6,sub7], 1, 0.5) 
            ts.registerCallback(self.callback_query)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([sub5,sub6], 1, 0.5) 
            ts.registerCallback(self.callback_query)

        transform = utils.create_transform_stamped((0,0,0),
                                            (0,0,0,1),
                                            rospy.Time.now(),
                                            query_odom_frame,
                                            self.map_frame_id) 

        self.frames_odom[query_frame_id] = query_odom_frame
        self.requests_timeout[query_frame_id] = requests_timeout
        self.prev_request[query_frame_id] = rospy.Time.now().to_sec()
        self.base_link[query_frame_id] = base_link
        self.N_msloc[query_frame_id] = N_msloc
        self.query_images[query_frame_id] = []
        self.query_poses[query_frame_id] = []
        self.retrieved_anchors[query_frame_id] = retrieved_anchors

        self.br.sendTransform(transform)        

    def camera_info_callback(self,msg):
        self.K1 = np.array(msg.K,dtype=np.float32).reshape(3,3)        

    def create_anchor(self, *args):
        try:
            if self.K1 is None:
                return

            if self.currently_running:
                return
            else:
                self.currently_running = True

            print('creating new anchor...')

            self.timestamp = args[0].header.stamp
            if self.robot_camera_frame_id is None:
                self.robot_camera_frame_id = args[0].header.frame_id
            try:
                self.ls.waitForTransform(self.map_frame_id,  self.robot_camera_frame_id, self.timestamp, rospy.Duration(self.tf_timout))
                pose1 = self.ls.lookupTransform(self.map_frame_id, self.robot_camera_frame_id, self.timestamp)
                pose1 = pose1[0]+pose1[1]
            except Exception as e:
                print('Error: '+str(e))
                self.currently_running = False   
                return

            cv_bridge = CvBridge()
            I1 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding=args[0].encoding)
            if args[0].encoding == 'rgb8':
                I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2BGR)   
            D1 = cv_bridge.imgmsg_to_cv2(args[1], desired_encoding='passthrough')
            K1 = self.K1

            self.ssloc.create_anchor(I1, D1, K1, pose1)

            self.currently_running = False        


        except Exception as e:
            self.currently_running = False        
            print('Error: '+str(e))
            return


    def callback_query(self,*args):
        
        try:
            query_frame_id = args[0].header.frame_id      
            timestamp_query = args[0].header.stamp     
            
            timeout = self.requests_timeout[query_frame_id]
            prev_request = self.prev_request[query_frame_id]

            N_msloc = self.N_msloc[query_frame_id]
            retrieved_anchors = self.retrieved_anchors[query_frame_id]

            # reduce timeout if images buffer > 0
            N_images = len(self.query_images[query_frame_id])                
            if N_images > 0:
                timeout = 1

            if prev_request + timeout > timestamp_query.to_sec() and timestamp_query.to_sec()>0:
                # print('request timeout..')
                return 
            else:
                self.prev_request[query_frame_id] = timestamp_query.to_sec()

            print('query image recieved from {}!'.format(query_frame_id))
            
            # unity_pose = self.ls.lookupTransform(self.query_camera_frame_id, 'unity', rospy.Time(0))
            unity_pose = utils.unpack_pose(args[2].pose)
            T_m2_c2 = utils.pq2matrix(unity_pose)

            # recalculate T_m2_c2 if the local SLAM pose uses base link frame instead of camera frame
            if self.base_link[query_frame_id]:
                T_link_c2 = utils.pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
                T_m2_link = T_m2_c2
                T_m2_c2 = T_m2_link.dot(T_link_c2)

            cv_bridge = CvBridge()
            # I2 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
            I2 = cv_bridge.compressed_imgmsg_to_cv2(args[0])         
            K2 = np.array(args[1].K,dtype=np.float32).reshape(3,3) 

            if N_msloc > 0:
                self.query_images[query_frame_id].append(I2)
                self.query_poses[query_frame_id].append(T_m2_c2)                
                N_images = len(self.query_images[query_frame_id])
                if N_images >= N_msloc:

                    T_m1_c2, len_best_inliers, T_m1_m2 = self.ssloc.callback_query_multiple(I2_l=self.query_images[query_frame_id], 
                                                                                    T2_l=self.query_poses[query_frame_id], 
                                                                                    K2=K2, optimization=False,
                                                                                    max_reproj_error=3,retrieved_anchors=retrieved_anchors,
                                                                                    second_inlier_ratio=0.0)
                    if T_m1_m2 is None:
                        return
                    
                    T_m2_m1 = utils.T_inv(T_m1_m2)                        

                    # clear images buffer
                    self.query_images[query_frame_id].clear()
                    self.query_poses[query_frame_id].clear()                    
                else:
                    return
            else:
                T_m1_c2, len_best_inliers = self.ssloc.callback_query(I2,K2,retrieved_anchors=retrieved_anchors,max_reproj_error=3)
                if T_m1_c2 is None:
                    return

                T_c2_m1 = utils.Tmatrix_inverse(T_m1_c2)
                T_m2_m1 = np.dot(T_m2_c2,T_c2_m1)

            # send localized pose relative to robot map
            self.send_unity2map_pose(T_m2_m1,query_frame_id,timestamp_query)            
            print('query camera localized!\n')  

        except Exception as e:
            print('Error: '+str(e))

    def send_unity2map_pose(self,T_m2_m1,query_frame_id,timestamp_query):
            

        # publish T_m2_m1 as posestamped message
        query_odom_frame = self.frames_odom[query_frame_id]

        t_unity = T_m2_m1[:3,3]
        q_unity = tf.transformations.quaternion_from_matrix(T_m2_m1)

        ps = PoseStamped()
        ps.header.stamp = timestamp_query
        ps.header.frame_id = query_odom_frame
        ps.pose.position.x = t_unity[0]
        ps.pose.position.y = t_unity[1]
        ps.pose.position.z = t_unity[2]
        ps.pose.orientation.x = q_unity[0]
        ps.pose.orientation.y = q_unity[1]
        ps.pose.orientation.z = q_unity[2]
        ps.pose.orientation.w = q_unity[3]

        self.pub.publish(ps)
        
        # np.savetxt(join(self.results_dir,'T_m2_m1.txt'),T_m2_m1)


if __name__ == '__main__':
    Node()
    