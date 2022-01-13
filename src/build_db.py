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
import csv
import json

hloc_module=os.path.join(os.path.dirname(os.path.realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import search_images

class Node:

    def __init__(self):
        rospy.init_node('build_db')

        self.dataset = os.path.join(hloc_module,'datasets')
        self.outputs = os.path.join(hloc_module,'outputs')
        self.image_dir = os.path.join(self.dataset,'images')

        self.map_id = rospy.get_param('~map_id',default='map')

        sub1 = message_filters.Subscriber('/image', Image)
        sub2 = message_filters.Subscriber('/camera_info', CameraInfo)

        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 1, 0.5) 
        ts.registerCallback(self.callback)

        rospy.Subscriber('trigger_build_db',String,self.callback_trigger)

        self.timeout = rospy.get_param('~timeout',default=30)
        self.ls = tf.TransformListener()

        self.pose_fname = os.path.join(self.dataset,'poses.csv')
        self.poses = []

        self.cam_fname = os.path.join(self.dataset,'intrinsics.json')
        self.K = []
        self.imsize = []
        self.D = []

        # create folders to save images
        try:
            os.mkdir(self.image_dir)
        except OSError as e:
            pass 

        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    def callback_trigger(self,*args):
        search_images.main(self.dataset,self.outputs,0,True)  

    def callback(self, *args):

        K = np.array(args[1].K,dtype=np.float32).reshape(3,3)
        t = args[0].header.stamp
        frame = args[0].header.frame_id
        seq = args[0].header.seq

        self.K = np.array(args[1].K,dtype=np.float32).reshape(3,3).tolist()
        self.imsize = [args[1].height,args[1].width]
        self.D = args[1].D

        cv_bridge = CvBridge()
        db_image = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough') 
        f_db = os.path.join(self.image_dir,str(seq)+'.jpg')
        cv2.imwrite(f_db,db_image)

        self.ls.waitForTransform(self.map_id, frame, t, rospy.Duration(self.timeout))
        try:
            p, q = self.ls.lookupTransform(self.map_id, frame, t)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        self.poses.append([seq]+p+q)


    def shutdown(self):
        # self.callback_trigger()
        with open(self.pose_fname, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for row in self.poses:
                writer.writerow(row)

        data = {"camera_matrix": self.K, "dist_coeff": self.D, "height": self.imsize[0], "width": self.imsize[1]}        
        with open(self.cam_fname, "w") as f:
            json.dump(data, f) 



if __name__ == '__main__':
    Node()    
    