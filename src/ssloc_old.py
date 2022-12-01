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

hloc_module=join(dirname(realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import match_features, detect_features, search

class Node:

    def __init__(self):
        
        self.timestamp = None
        self.timestamp_query = None
        self.query_camera_frame_id = None
        self.timeout = 1.5
        self.counter = 0
        
        self.K1 = None
        self.currently_running = False
        self.T_m1_c2_buffer = []
        self.T_m2_c2_buffer = []
        self.T_m2_m1_current = None
        self.frames_odom = dict()
        self.requests_timeout = dict()
        self.prev_request = dict()
        self.use_camera_coords = dict()

        # initialize parameters
        rospy.init_node('self_localization')
        self.map_frame_id = rospy.get_param('~map_frame_id',default='map')
        self.robot_camera_frame_id = rospy.get_param('~camera_frame_id',default=None)
        self.send_unity_pose = rospy.get_param('~send_unity_pose',default=False)        
        self.frame_rate = rospy.get_param('~frame_rate',default=0.5)        
        self.detector = rospy.get_param('~detector',default='SuperPoint')
        self.matcher = rospy.get_param('~matcher',default='SuperGlue')  
        self.sliding_average_buffer = rospy.get_param('~sliding_average_buffer',default=1)       
        self.num_retrieved_anchors = rospy.get_param('~num_retrieved_anchors',default=5)        

        self.results_dir = rospy.get_param('~save_directory',default='results')
        self.create_new_anchors = rospy.get_param('~create_new_anchors',default=False)        
        self.num_query_devices = rospy.get_param('/num_users',default=1) 

        # load models
        self.matcher_model = None
        self.detector_model = None
        self.retrieval_model = None
        self.load_models()

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
            timeout = 0
            self.add_player_subs(image_query,camera_info_query,pose_query,query_frame_id,query_odom_frame)

        # add subscribers for tello drone
        image_query = "/tello/camera/image_raw/compressed"
        camera_info_query = "/tello/camera/camera_info"
        pose_query = "/tello/real_world_pos"
        query_frame_id = 'tello_camera'
        query_odom_frame = 'tello_camera_odom'
        timeout = 60
        use_camera_coords = False
        self.add_player_subs(image_query,camera_info_query,pose_query,query_frame_id,query_odom_frame,timeout,use_camera_coords)

        self.pub = rospy.Publisher('reloc_map_pose', PoseStamped, queue_size=1)
        self.pub2 = rospy.Publisher('retrieved_image', Image, queue_size=1)
        self.pub3 = rospy.Publisher('matches_image', Image, queue_size=1)

        rospy.spin()

    def add_player_subs(self,image_query,camera_info_query,pose_query,query_frame_id,query_odom_frame,timeout=0,use_camera_coords=True):
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
        self.requests_timeout[query_frame_id] = timeout
        self.prev_request[query_frame_id] = rospy.Time.now().to_sec()
        self.use_camera_coords[query_frame_id] = use_camera_coords

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
                self.ls.waitForTransform(self.map_frame_id,  self.robot_camera_frame_id, self.timestamp, rospy.Duration(self.timeout))
                pose1 = self.ls.lookupTransform(self.map_frame_id, self.robot_camera_frame_id, self.timestamp)
                pose1 = pose1[0]+pose1[1]
            except Exception as e:
                print('Error: '+e)
                self.currently_running = False   
                return

            cv_bridge = CvBridge()
            I1 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding=args[0].encoding)
            if args[0].encoding == 'rgb8':
                I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2BGR)   
            D1 = cv_bridge.imgmsg_to_cv2(args[1], desired_encoding='passthrough')
            K1 = self.K1

            # detect local features
            fname1 = join(self.results_dir,'local_features','local_%i.h5' % self.counter)
            kp1, des1 = self.feature_detection(I1, self.detector, fname1, model=self.detector_model)

            # detect global features
            fname1 = join(self.results_dir,'global_features','global_%i.h5' % self.counter)
            self.feature_detection(I1, 'netvlad', fname1, model=self.retrieval_model, id='%i.jpg' % self.counter)

            # save rgb image
            fname_rgb1 = join(self.results_dir,'rgb','rgb_%i.png' % self.counter)      
            cv2.imwrite(fname_rgb1,I1)

            # save depth image
            fname_depth1 = join(self.results_dir,'depth','depth_%i.png' % self.counter)
            D1 = np.array(D1,dtype=np.uint16)
            D1[D1 > 65535] = 65535        
            cv2.imwrite(fname_depth1,D1)

            # save camera info
            fname_info1 = join(self.results_dir,'K1.txt')
            np.savetxt(fname_info1,K1)

            # save pose
            fname_pose1 = join(self.results_dir,'poses','pose_%i.txt' % self.counter)
            np.savetxt(fname_pose1,pose1)

            print('created anchor %i!' % self.counter)

            rate = rospy.Rate(self.frame_rate)
            
            self.counter += 1
            
            self.currently_running = False        


        except Exception as e:
            self.currently_running = False        
            print('Error: '+e)
            return


    def callback_query(self,*args):
        print('query image recieved!')
        try:
            query_frame_id = args[0].header.frame_id      
            timestamp_query = args[0].header.stamp     

            timeout = self.requests_timeout[query_frame_id]
            prev_request = self.prev_request[query_frame_id]

            if prev_request + timeout > timestamp_query.to_sec():
                print('request timeout..')
                return 
            else:
                self.prev_request[query_frame_id] = timestamp_query.to_sec()

            if self.send_unity_pose:
                # unity_pose = self.ls.lookupTransform(self.query_camera_frame_id, 'unity', rospy.Time(0))
                unity_pose = utils.unpack_pose(args[2].pose)

            cv_bridge = CvBridge()
            # I2 = cv_bridge.imgmsg_to_cv2(args[0], desired_encoding='passthrough')         
            I2 = cv_bridge.compressed_imgmsg_to_cv2(args[0])         
            K2 = np.array(args[1].K,dtype=np.float32).reshape(3,3) 


            print('Detecting local features in query image...')
            fname2_local = join(self.results_dir,'local_features','local_q.h5')        
            kp2, des2 = self.feature_detection(I2, self.detector, fname2_local, model=self.detector_model)
            
            print('Retrieving similar anchors...')
            fdir_db = join(self.results_dir,'global_features')
            fname2_global = join(self.results_dir,'global_q.h5')
            self.feature_detection(I2, 'netvlad', fname2_global, model=self.retrieval_model, id='q.jpg')

            pairs,scores = search.main(fname2_global,fdir_db,num_matches=20)
            
            # print('\nsimilarity score between anchor and query: %.4f' % scores[0])
            pts2D_all = np.array([]).reshape(0,2)
            pts3D_all = np.array([]).reshape(0,3)

            matches1 = []
            ret_index1 = None

            for i,(p,s) in enumerate(zip(pairs,scores)):

                if i > self.num_retrieved_anchors:
                    break

                if s < 0.1:
                    continue

                ret_index = int(p[1].replace('.jpg',''))
                print('retrieved anchor %i\n' % ret_index)

                # load rgb image
                I1 = cv2.imread(join(self.results_dir,'rgb','rgb_%i.png' % ret_index))
                D1 = cv2.imread(join(self.results_dir,'depth','depth_%i.png' % ret_index),cv2.IMREAD_UNCHANGED)
                K1 = np.loadtxt(join(self.results_dir,'K1.txt'))
                pose1 = np.loadtxt(join(self.results_dir,'poses','pose_%i.txt' % ret_index))

                self.pub2.publish(cv_bridge.cv2_to_imgmsg(I1, encoding='passthrough'))

                fname1_local = join(self.results_dir,'local_features','local_%i.h5' % ret_index)             
                kp1 = detect_features.load_features(fname1_local)
                des1 = None # not used for superpoint

                print('Matching features in query image...')
                matches = self.feature_matching(des1,des2,self.detector,self.matcher, fname1_local, fname2_local, model=self.matcher_model)

                img_matches = self.draw_matches_ros(I1,I2,kp1,kp2,matches)

                self.pub3.publish(cv_bridge.cv2_to_imgmsg(img_matches, encoding='passthrough'))             

                if len(matches) >len(matches1):
                    matches1 = matches
                    ret_index1 = ret_index

                if len(matches) > 10:
                    pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
                    pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

                    x_c,y_c,z_c = utils.project_2d_to_3d(pts1.T,K1,D1,h=0)

                    pts3D_c = np.array([x_c,y_c,z_c,np.ones(x_c.shape[0])])

                    tc = pose1[:3]
                    qc = pose1[3:]
                    T_m1_c1 = tf.transformations.quaternion_matrix(qc)
                    T_m1_c1[:3,3] = tc

                    pts3D = T_m1_c1.dot(pts3D_c)
                    pts3D = pts3D[:3,:]/pts3D[3,:]

                    pts3D = pts3D.T            
                    
                    idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
                    if len(idx) == 0:
                        break

                    pts3D = pts3D[idx]
                    pts2D = pts2[idx]

                    pts2D_all = np.vstack([pts2D_all,pts2D])
                    pts3D_all = np.vstack([pts3D_all,pts3D])

            if pts2D_all.shape[0] < 10:
                print('\nNo anchors found! Try again with another query image..\n')  
                return

            retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D_all, pts2D_all, K2, None,flags=cv2.SOLVEPNP_P3P)
            
            # find relocalized pose of query image relative to robot camera
            R_ = cv2.Rodrigues(rvecs)[0]
            R = R_.T
            C = -R_.T.dot(tvecs)  

            # send localized pose relative to robot map
            T_m1_c2=self.send_reloc_pose(C,R,query_frame_id,timestamp_query)
            if self.send_unity_pose:
                self.send_unity2map_pose(unity_pose,T_m1_c2,query_frame_id,timestamp_query)            

            print('\nquery camera localized!\n')  

        except Exception as e:
            print('Error: '+e)


    def load_anchor(kp1, D1, K1, pose1):
        pts1 = np.float32([ kp.pt for kp in kp1 ])
        x_c,y_c,z_c = utils.project_2d_to_3d(pts1.T,K1,D1)
        pts3D_c = np.array([x_c,y_c,z_c,np.ones(x_c.shape[0])])

        tc = pose1[:3]
        qc = pose1[3:]
        T_m1_c1 = tf.transformations.quaternion_matrix(qc)
        T_m1_c1[:3,3] = tc

        pts3D_m = T_m1_c1.dot(pts3D_c)
        pts3D_m = pts3D_m[:3,:]/pts3D_m[3,:]

        pts3D_m = pts3D_m.T  

        valid = np.array([i for i,p in enumerate(pts3D_m) if not np.any(np.isnan(p))])
        pts3D_m = pts3D_m[valid]
        kp1 = [kp1[v] for v in valid]
        des1 = des1[valid,:]    

        return pts3D_m           

    def send_reloc_pose(self,C,R,query_frame_id,timestamp_query):
        R2 = np.eye(4)
        R2[:3,:3] = R
        q = tf.transformations.quaternion_from_matrix(R2)     
        transform = utils.create_transform_stamped((C[0],C[1],C[2]),
                                            (q[0],q[1],q[2],q[3]),
                                            timestamp_query,
                                            query_frame_id,
                                            self.map_frame_id)

        # self.br.sendTransform(transform)
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3,:3] = R
        T_m1_c2[:3,3] = C.reshape(-1)        
        self.T_m1_c2_buffer.append(T_m1_c2)
        if len(self.T_m2_c2_buffer) > self.sliding_average_buffer:
            self.T_m2_c2_buffer.pop(0)

        return T_m1_c2

    def send_unity2map_pose(self,unity_pose,T_m1_c2,query_frame_id,timestamp_query):
            
        if not self.use_camera_coords[query_frame_id]:
            T_link_c2 = utils.pose2matrix([0,0,0,-0.5,0.5,-0.5,0.5])
            T_c2_link = utils.T_inv(T_link_c2)
            T_m1_link = T_m1_c2.dot(T_c2_link)
            # declare c2 to be link frame
            T_m1_c2 = T_m1_link
            
        (t,q) = unity_pose                    

        R = tf.transformations.quaternion_matrix(q)[:3,:3]
        T_m2_c2 = np.eye(4)
        T_m2_c2[:3,:3] = R
        T_m2_c2[:3,3] = t
        self.T_m2_c2_buffer.append(T_m2_c2)
        if len(self.T_m2_c2_buffer) > self.sliding_average_buffer:
            self.T_m2_c2_buffer.pop(0)

        R_inv = R.T
        t_inv = -R_inv.dot(t)
        R2 = np.eye(4)
        R2[:3,:3] = R_inv
        q_inv = tf.transformations.quaternion_from_matrix(R2) 

        query_odom_frame = self.frames_odom[query_frame_id]

        transform = utils.create_transform_stamped((t_inv[0],t_inv[1],t_inv[2]),
                              (q_inv[0],q_inv[1],q_inv[2],q_inv[3]),
                               timestamp_query,
                               query_odom_frame,
                               query_frame_id)

        # self.br.sendTransform(transform)

        # 1 element
        # if len(self.T_m1_c2_buffer) == 1:
        T_c2_m1 = utils.Tmatrix_inverse(T_m1_c2)
        T_m2_m1 = np.dot(T_m2_c2,T_c2_m1)
        self.T_m2_m1_current = T_m2_m1    
        # else:
            # implement method to include previous localization results

        t_unity = self.T_m2_m1_current[:3,3]
        q_unity = tf.transformations.quaternion_from_matrix(self.T_m2_m1_current)

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
        
        np.savetxt(join(self.results_dir,'T_m2_m1.txt'),T_m2_m1)

    def draw_matches_ros(self,I1,I2,kp1,kp2,matches):
        img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
        return img
                           

    def feature_detection(self,I,detector,fname=None,model=None,id=None):
        
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
            kp = detect_features.main(I,fname,detector,model=model,id=id)  
            des = None        

        return kp, des

    def feature_matching(self,des1,des2,detector,matcher,fname1=None,fname2=None, model=None):

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
            matches = match_features.main(fname1,fname2,detector,matcher,model=model)

        return matches    

    def load_models(self):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            return     
        else:
            self.matcher_model = match_features.load_model(self.detector,self.matcher)
            self.detector_model = detect_features.load_model(self.detector)        

        self.retrieval_model = detect_features.load_model('netvlad')        
        print('Loaded Netvlad model')

  

if __name__ == '__main__':
    Node()
    