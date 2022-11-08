#!/usr/bin/env python3
import numpy as np
import sys
import os
from os.path import join, dirname, realpath
import cv2
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch
import pickle
import open3d as o3d
import copy
from datetime import datetime
import time
import json

hloc_module = join(dirname(realpath(__file__)), 'hloc_toolbox')
sys.path.insert(0, hloc_module)

from hloc_toolbox import match_features, detect_features, search
from hloc_toolbox.hloc.matchers import loftr


torch.cuda.empty_cache()


class Node:

    def __init__(self, debug=False, data_folder=None, create_new_anchors=False, detector='SuperPoint', matcher='SuperGlue'):

        self.debug = debug
        self.map_frame_id = None
        self.robot_camera_frame_id = None
        self.send_unity_pose = False
        self.frame_rate = 1
        self.detector = detector
        self.matcher = matcher
        self.max_retrieved_anchors = 5
        self.similarity_threshold = 0.1
        self.sliding_average_buffer = 1
        self.results_dir = join(data_folder, 'anchors')
        self.data_folder = data_folder
        self.create_new_anchors = create_new_anchors

        self.K1 = None
        self.currently_running = False
        self.T_m1_c2_buffer = []
        self.T_m2_c2_buffer = []
        self.T_m2_m1_current = None

        self.matcher_model = None
        self.detector_model = None
        self.retrieval_model = None
        self.load_models()

        # utils.make_dir(join(self.results_dir), delete_if_exists=self.create_new_anchors)
        utils.make_dir(join(self.results_dir, 'local_features'))
        utils.make_dir(join(self.results_dir, 'global_features'))
        utils.make_dir(join(self.results_dir, 'rgb'))
        utils.make_dir(join(self.results_dir, 'depth'))
        utils.make_dir(join(self.results_dir, 'poses'))

    def create_anchor(self, *args):
        try:
            if self.K1 is None:
                return

            # if self.currently_running:
            #     return
            # else:
            #     self.currently_running = True

            print('creating new anchor...')

            I1 = args[0]
            K1 = self.K1
            D1 = args[1]
            pose1 = args[2]

            # detect local features
            fname1 = join(self.results_dir, 'local_features', 'local_%i.h5' % self.counter)
            kp1, des1 = self.feature_detection(I1, self.detector, fname1, model=self.detector_model)

            # detect global features
            fname1 = join(self.results_dir, 'global_features', 'global_%i.h5' % self.counter)
            self.feature_detection(I1, 'netvlad', fname1, model=self.retrieval_model, id='%i.jpg' % self.counter)

            # save rgb image
            fname_rgb1 = join(self.results_dir, 'rgb', 'rgb_%i.png' % self.counter)
            cv2.imwrite(fname_rgb1, I1)

            # save depth image
            fname_depth1 = join(self.results_dir, 'depth', 'depth_%i.png' % self.counter)
            D1 = np.array(D1, dtype=np.uint16)
            D1[D1 > 65535] = 65535
            cv2.imwrite(fname_depth1, D1)

            # save camera info
            fname_info1 = join(self.results_dir, 'K1.txt')
            np.savetxt(fname_info1, K1)

            # save pose
            fname_pose1 = join(self.results_dir, 'poses', 'pose_%i.txt' % self.counter)
            np.savetxt(fname_pose1, pose1)

            print('created anchor %i!' % self.counter)


        except Exception as e:
            self.currently_running = False
            print(e)
            return

    def callback_query(self, I2, K2, max_reproj_error=None,fov=None):

        if max_reproj_error is None:
            max_reproj_error = 8

        if K2 is None:
            I2_l, _, K2 = utils.fun_rectify_views(I2, fov)            
            I2 = I2_l[0]

        print('query image recieved!')

        print('Detecting local features in query image...')
        fname2_local = join(self.results_dir, 'local_features', 'local_q.h5')
        kp2, des2 = self.feature_detection(I2, self.detector, fname2_local, model=self.detector_model)

        print('Retrieving similar anchors...')
        fdir_db = join(self.results_dir, 'global_features')
        fname2_global = join(self.results_dir, 'global_q.h5')
        self.feature_detection(I2, 'netvlad', fname2_global, model=self.retrieval_model, id='q.jpg')

        pairs, scores = search.main(fname2_global, fdir_db, num_matches=self.max_retrieved_anchors)

        print('\nsimilarity score between anchor and query: %.4f' % scores[0])
        pts2D_all = np.array([]).reshape(0, 2)
        pts3D_all = np.array([]).reshape(0, 3)

        matches1 = []
        ret_index1 = None

        for i, (p, s) in enumerate(zip(pairs, scores)):

            if i > self.max_retrieved_anchors:
                break  # terminate loop

            if s < self.similarity_threshold:
                continue  # skip the current iteration of the loop

            ret_index = int(p[1].replace('.jpg', ''))
            print('retrieved anchor %i\n' % ret_index)

            # load rgb image
            I1 = cv2.imread(join(self.results_dir, 'rgb', 'rgb_%i.png' % ret_index))
            D1 = cv2.imread(join(self.results_dir, 'depth', 'depth_%i.png' % ret_index), cv2.IMREAD_UNCHANGED)
            K1 = np.loadtxt(join(self.results_dir, 'K1.txt'))
            pose1 = np.loadtxt(join(self.results_dir, 'poses', 'pose_%i.txt' % ret_index))
            # if self.ros:
            #     self.pub2.publish(cv_bridge.cv2_to_imgmsg(I1, encoding='passthrough'))

            if self.detector == "loftr":
                des1 = None
                des2 = None
                fname1_local = None
                fname2_local = None
            else:
                fname1_local = join(self.results_dir, 'local_features', 'local_%i.h5' % ret_index)
                kp1 = detect_features.load_features(fname1_local)
                des1 = None  # not used for superpoint

            print('Matching features in query image...')
            if self.matcher == "loftr":
                matches, kp1, kp2 = self.feature_matching(des1, des2, self.detector, self.matcher, fname1_local,
                                                          fname2_local,
                                                          model=self.matcher_model, img1=I1, img2=I2)
            else:
                matches = self.feature_matching(des1, des2, self.detector, self.matcher, fname1_local, fname2_local,
                                                model=self.matcher_model, img1=I1, img2=I2)

            if len(matches) > len(matches1):
                matches1 = matches
                ret_index1 = ret_index

            if len(matches) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                x_c, y_c, z_c = utils.project_2d_to_3d(pts1.T, K1, D1, h=0)

                pts3D_c = np.array([x_c, y_c, z_c, np.ones(x_c.shape[0])])

                tc = pose1[:3]
                qc = pose1[3:]
                # T_m1_c1 = tf.transformations.quaternion_matrix(qc)
                T_m1_c1 = np.eye(4)
                T_m1_c1[:3, :3] = Rotation.from_quat(qc).as_matrix()
                T_m1_c1[:3, 3] = tc

                # T_m1_c1[:3, 3] = tc

                pts3D = T_m1_c1.dot(pts3D_c)
                pts3D = pts3D[:3, :] / pts3D[3, :]

                pts3D = pts3D.T

                idx = np.array([i for i, p in enumerate(pts3D) if not np.any(np.isnan(p))])
                if len(idx) == 0:
                    break

                pts3D = pts3D[idx]
                pts2D = pts2[idx]

                pts2D_all = np.vstack([pts2D_all, pts2D])
                pts3D_all = np.vstack([pts3D_all, pts3D])

        if pts2D_all.shape[0] < 10:
            print('\nNo anchors found! Try again with another query image..\n')
            return None,None

        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3D_all, pts2D_all, K2, None, flags=cv2.SOLVEPNP_P3P, reprojectionError=max_reproj_error, iterationsCount=10000)

        # find relocalized pose of query image relative to robot camera
        R_ = cv2.Rodrigues(rvecs)[0]
        R = R_.T
        C = -R_.T.dot(tvecs)
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3, :3] = R
        T_m1_c2[:3, 3] = C.reshape(-1)

        num_inliers = len(inliers)

        return T_m1_c2, num_inliers

    def callback_query_multiple(self, Ip=None, fov=90, I2_l=None, T2_l=None, K2=None, 
                                optimization=True, max_reproj_error=10, one_view=None):
        
        if Ip is not None:
            print('query panorama image recieved!')
            I2_l, T2_l, K2 = utils.fun_rectify_views(Ip, fov)
        elif I2_l is not None and T2_l is not None and K2 is not None:
            print('multiple query images recieved!')
        else:
            raise Exception("incorrect inputs provided to function")

        query_poses_l = []
        pts2D_all = []
        pts3D_all = []
        N = len(I2_l)
        max_retries = 4

        # print('\nsimilarity score between anchor and query: %.4f' % scores[0])

        for k in range(N):

            pts2D_img = np.array([]).reshape(0,2)
            pts3D_img = np.array([]).reshape(0,3)                                

            print('\nProcessing view %i\n' % (k+1))
            I2 = I2_l[k]

            # query_rot = tf.transformations.quaternion_from_matrix(T2_l[k]).tolist()
            query_rot = Rotation.from_matrix(T2_l[k][:3, :3]).as_quat().tolist()
            query_pos = T2_l[k][:3,3].tolist()
            query_poses_l.append(query_pos + query_rot)

            if one_view is not None:
                if k+1 != one_view:
                    pts2D_all.append(pts2D_img)
                    pts3D_all.append(pts3D_img)  
                    continue 

            print('Detecting local features in query image...')
            fname2_local = join(self.results_dir,'local_features','local_q.h5')        
            kp2, des2 = self.feature_detection(I2, self.detector, fname2_local, model=self.detector_model)
            
            print('Retrieving similar anchors...')
            fdir_db = join(self.results_dir,'global_features')
            fname2_global = join(self.results_dir,'global_q.h5')
            self.feature_detection(I2, 'netvlad', fname2_global, model=self.retrieval_model, id='q.jpg')

            pairs,scores = search.main(fname2_global,fdir_db,num_matches=self.max_retrieved_anchors)
            
            matches1 = []
            ret_index1 = None

            for i,(p,s) in enumerate(zip(pairs,scores)):

                if i > self.max_retrieved_anchors:
                    break

                if s < self.similarity_threshold:
                    continue

                ret_index = int(p[1].replace('.jpg',''))
                print('retrieved anchor %i\n' % ret_index)

                # load rgb image
                I1 = cv2.imread(join(self.results_dir,'rgb','rgb_%i.png' % ret_index))
                D1 = cv2.imread(join(self.results_dir,'depth','depth_%i.png' % ret_index),cv2.IMREAD_UNCHANGED)
                K1 = np.loadtxt(join(self.results_dir,'K1.txt'))
                pose1 = np.loadtxt(join(self.results_dir,'poses','pose_%i.txt' % ret_index))

                if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
                    fname1_local = join(self.results_dir,'rgb','rgb_%i.png' % ret_index)    
                    I1 = cv2.imread(fname1_local)  
                    kp1, des1 = self.feature_detection(I1, self.detector)       
                elif self.detector == "loftr":
                    des1 = None
                    des2 = None
                    fname2_local = None                
                    fname1_local = join(self.results_dir,'rgb','rgb_%i.png' % ret_index)    
                    I1 = cv2.imread(fname1_local)  
                else:
                    fname1_local = join(self.results_dir,'local_features','local_%i.h5' % ret_index)             
                    kp1 = detect_features.load_features(fname1_local)
                    des1 = None # not used for superpoint

                print('Matching features in query image...')


                print('Matching features in query image...')
                if self.matcher == "loftr":
                    matches, kp1, kp2 = self.feature_matching(des1, des2, self.detector, self.matcher, fname1_local,
                                                            fname2_local,
                                                            model=self.matcher_model, img1=I1, img2=I2)
                else:
                    matches = self.feature_matching(des1,des2,self.detector,self.matcher, fname1_local, fname2_local, 
                                                    model=self.matcher_model)


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
                    T_m1_c1 = np.eye(4)
                    T_m1_c1[:3, :3] = Rotation.from_quat(qc).as_matrix()
                    T_m1_c1[:3,3] = tc

                    pts3D = T_m1_c1.dot(pts3D_c)
                    pts3D = pts3D[:3,:]/pts3D[3,:]

                    pts3D = pts3D.T            
                    
                    idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
                    if len(idx) == 0:
                        break

                    pts3D = pts3D[idx]
                    pts2D = pts2[idx]

                    pts2D_img = np.vstack([pts2D_img,pts2D])
                    pts3D_img = np.vstack([pts3D_img,pts3D])

            pts2D_all.append(pts2D_img)
            pts3D_all.append(pts3D_img)

        if np.sum([len(p) for p in pts2D_all]) < 50:
            print('\nNo anchors found! Try again with another query image..\n')  
            return None, None, None

        if one_view is not None:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3D_all[0], pts2D_all[0], K2, None, flags=cv2.SOLVEPNP_P3P)
            best_inlier_idxs = [inliers]
            rvecs_l = [rvecs]
            tvecs_l = [tvecs]
        else:
            for r in range(max_retries):
                
                T_m1_m2,tvecs_l,rvecs_l,best_inlier_idxs=utils.multiviewSolvePnPRansac(pts3D_all, pts2D_all, query_poses_l, K2, max_reproj_error=max_reproj_error) 
                
                len_best_inlier_idxs = sorted([len(inliers) for inliers in best_inlier_idxs], reverse=True)
                if len_best_inlier_idxs[1]/len_best_inlier_idxs[0] > 0.15:
                    break
                else:
                    print('Relaxing re-projection error threshold...')
                    max_reproj_error += 5
                
        if optimization:
            pts2D_all = [p[inliers] for p,inliers in zip(pts2D_all,best_inlier_idxs)]
            pts3D_all = [p[inliers] for p,inliers in zip(pts3D_all,best_inlier_idxs)]
            T_m1_m2,tvecs_l,rvecs_l=utils.multiviewSolvePnPOptimization(pts3D_all, pts2D_all, query_poses_l, K2, T_m1_m2) 

        print('\nquery cameras localized!\n')  

        len_best_inlier_idxs = [len(inliers) for inliers in best_inlier_idxs]
        if one_view is not None:
            print('Inliers in View {}: {}' .format(one_view, len(inliers)))
        else:
            for i in range(len(best_inlier_idxs)):
                print('Inliers in View {:d}: {:d}' .format(i, len(best_inlier_idxs[i])))

        # compute relocalized pose of front image
        R_ = cv2.Rodrigues(rvecs_l[0])[0]
        R = R_.T
        C = -R_.T.dot(tvecs_l[0])
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3, :3] = R
        T_m1_c2[:3, 3] = C.reshape(-1)
        if one_view:
            T_m1_m2 = T_m1_c2

        return T_m1_c2, len_best_inlier_idxs, T_m1_m2

    def feature_detection(self, I, detector, fname=None, model=None, id=None):

        if detector == 'SIFT':
            gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
        elif detector == 'ORB':
            orb = cv2.ORB_create(nfeatures=5000)
            kp, des = orb.detectAndCompute(I, None)
        elif detector == 'SURF':
            surf = cv2.xfeatures2d.SURF_create()
            kp, des = surf.detectAndCompute(I, None)
        elif detector == "loftr":
            return None, None
        else:
            if os.path.exists(fname):
                os.remove(fname)
            kp = detect_features.main(I, fname, detector, model=model, id=id)
            des = None

        return kp, des

    def feature_matching(self, des1, des2, detector, matcher, fname1=None, fname2=None, model=None, img1=None,
                         img2=None):

        if detector == 'ORB' or detector == 'SIFT' or detector == 'SURF':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            matches = good
            return matches
        elif detector == "loftr":

            s1 = 1024 / img1.shape[1]
            s2 = 1024 / img2.shape[1] 

            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            # img1 = cv2.resize(img1,
            #                   (img1.shape[1] // 8 * 8, img1.shape[0] // 8 * 8))  # input size shuold be divisible by 8
            # img2 = cv2.resize(img2, (img2.shape[1] // 8 * 8, img2.shape[0] // 8 * 8))

            img1 = cv2.resize(img1, (int(img1.shape[1] * s1), int(img1.shape[0] * s1)))
            img2 = cv2.resize(img2, (int(img2.shape[1] * s2), int(img2.shape[0] * s2)))

            img1 = torch.from_numpy(img1)[None][None].cuda() / 255.
            img2 = torch.from_numpy(img2)[None][None].cuda() / 255.
            batch = {'image0': img1, 'image1': img2}

            matches = model(batch)

            """
            Need to go from matches which is a dictionary of torch tensors to opencv matches
            https://docs.opencv2.org/3.4/d4/de0/classcv_1_1DMatch.html#ab9800c265dcb748a28aa1a2d4b45eea4
            """

            mkpts0 = matches['mkpts0_f'].cpu().numpy()
            mkpts1 = matches['mkpts1_f'].cpu().numpy()

            matches = [None] * int(mkpts0.shape[0])
            kp1 = [None] * int(mkpts0.shape[0])
            kp2 = [None] * int(mkpts0.shape[0])
            for i in range(mkpts0.shape[0]):
                matches[i] = cv2.DMatch(i, i, 0)
                kp1[i] = cv2.KeyPoint(mkpts0[i][0]/s1, mkpts0[i][1]/s1, 0, 1, -1)
                kp2[i] = cv2.KeyPoint(mkpts1[i][0]/s2, mkpts1[i][1]/s2, 0, 1, -1)

            return matches, kp1, kp2
        else:
            matches = match_features.main(fname1, fname2, detector, matcher, model=model)

        return matches

    def load_models(self):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            return
        elif self.detector == "loftr" or self.detector == "loftr":
            default_conf = {
                'weights': 'outdoor_ds.ckpt',
                'max_num_matches': 5000,
            }
            self.matcher_model = loftr.loftr(default_conf)
        else:
            self.matcher_model = match_features.load_model(self.detector, self.matcher)
            self.detector_model = detect_features.load_model(self.detector)

        self.retrieval_model = detect_features.load_model('netvlad')
        print('Loaded Netvlad model')

    def draw_matches(self, I1, I2, kp1, kp2, matches, fname, PLOT_FIGS=False):

        img = cv2.drawMatches(I1, kp1, I2, kp2, matches, None, flags=2)
        if PLOT_FIGS:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), plt.show()

        cv2.imwrite(fname, img)

    def create_offline_anchors(self,skip=None,num_images=None):
        image_dir = join(self.data_folder, 'rgb')
        poses = np.loadtxt(join(self.data_folder, 'poses.csv'), delimiter=",")
        
        if num_images is None:
            num_images = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])        

        if os.path.exists(join(self.data_folder, 'K1.txt')):
            K1 = np.loadtxt(join(self.data_folder, 'K1.txt'))
        elif os.path.exists(join(self.data_folder, 'intrinsics.json')):
            with open(join(self.data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K1 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')
        self.K1 = K1

        for i in range(num_images):
            if skip is not None and i % skip != 0:
                continue
            
            I1 = cv2.imread(join(self.data_folder, 'rgb', str(i + 1) + '.png'))
            D1 = cv2.imread(join(self.data_folder, 'depth', str(i + 1) + '.png'), cv2.IMREAD_UNCHANGED)
            pose1 = poses[i][1:8]
            self.counter = i
            self.create_anchor(I1, D1, pose1)
            print('\nAnchor created for image %i/%i\n' % (i + 1, num_images))


    def query_panoramas(self, query_data_folder, optimization=True, one_view=False, results_prefix='', max_reproj_error=10):

        # query_img_dir = join(query_data_folder, 'rgb')
        query_img_dir = query_data_folder

        poses = np.empty((0, 7))
        num_inliers = []

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)
            if one_view:
                K2=None
                T_m1_c2, len_best_inliers = self.callback_query(I2,K2,fov=90)
            else:
                T_m1_c2, len_best_inliers, _ = self.callback_query_multiple(Ip=I2, optimization=optimization, max_reproj_error=max_reproj_error)
            if T_m1_c2 is not None:
                R = Rotation.from_matrix(T_m1_c2[:3, :3])
                q = R.as_quat()
                t = T_m1_c2[:3, 3].T
                pose = np.concatenate((t, q), axis=0)
                poses = np.vstack((poses, pose))
                num_inliers.append(len_best_inliers)
            else:
                print("No pose found for image {}".format(filename))
                er = [0,0,0,0,0,0,1]
                poses = np.vstack((poses, er)) 
                if one_view:
                    num_inliers.append(0)                       
                else:
                    num_inliers.append([0, 0, 0, 0])                       

        np.savetxt(join(query_data_folder, results_prefix+'poses.csv'), poses, delimiter=',')
        np.savetxt(join(query_data_folder, results_prefix+'num_inliers.csv'), num_inliers, delimiter=',', fmt='%i')

    def query_front(self, query_data_folder, max_reproj_error=None):

        # query_img_dir = join(query_data_folder, 'rgb')
        query_img_dir = query_data_folder

        poses = np.empty((0, 7))
        num_inliers = []

        if os.path.exists(join(query_data_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_data_folder, 'K2.txt'))
        elif os.path.exists(join(query_data_folder, 'intrinsics.json')):
            with open(join(query_data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')        

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)
            T_m1_c2, len_best_inliers = self.callback_query(I2,K2,max_reproj_error=max_reproj_error)
            if T_m1_c2 is not None:
                R = Rotation.from_matrix(T_m1_c2[:3, :3])
                q = R.as_quat()
                t = T_m1_c2[:3, 3].T
                pose = np.concatenate((t, q), axis=0)
                poses = np.vstack((poses, pose))
                num_inliers.append(len_best_inliers)
            else:
                print("No pose found for image {}".format(filename))
                poses = np.vstack((poses, np.zeros(7))) 
                num_inliers.append(0)                       

        np.savetxt(join(query_data_folder, 'poses.csv'), poses, delimiter=',')
        np.savetxt(join(query_data_folder, 'num_inliers.csv'), num_inliers, delimiter=',', fmt='%i')


    def query_front_multiple(self, query_data_folder, max_reproj_error=10, optimization=False):

        # query_img_dir = join(query_data_folder, 'rgb')
        query_img_dir = query_data_folder


        if os.path.exists(join(query_data_folder, 'K2.txt')):
            K2 = np.loadtxt(join(query_data_folder, 'K2.txt'))
        elif os.path.exists(join(query_data_folder, 'intrinsics.json')):
            with open(join(query_data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            K2 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')  

        I2_l = []
        T2_l = []
        query_poses_l = np.loadtxt(join(query_data_folder, 'poses_local.csv'), delimiter=',')

        for num, filename in enumerate(sorted(os.listdir(query_img_dir))):
            ext = filename.split('.')[-1]
            if not (ext == 'jpg' or ext == 'png' or ext == 'JPG'):
                continue
            id = int(filename.split('.')[0])
            image_path = os.path.join(query_img_dir, filename)
            I2 = cv2.imread(image_path)
            I2_l.append(I2)
            p = query_poses_l[id-1,1:]
            T_m2_c2 = np.eye(4)
            T_m2_c2[:3,:3] = Rotation.from_quat(p[3:]).as_matrix()
            T_m2_c2[:3,3] = p[:3]
            T2_l.append(T_m2_c2)

        _, len_best_inliers, T_m1_m2 = self.callback_query_multiple(I2_l=I2_l, T2_l=T2_l, 
                                                                 K2=K2, optimization=optimization,
                                                                 max_reproj_error=max_reproj_error)
        poses = np.empty((0, 7))
        for i in range(len(T2_l)):

            T_m2_c2 = T2_l[i]
            T_m1_c2 = T_m1_m2.dot(T_m2_c2)

            R = Rotation.from_matrix(T_m1_c2[:3, :3])
            q = R.as_quat()
            t = T_m1_c2[:3, 3].T
            pose = np.concatenate((t, q), axis=0)
            poses = np.vstack((poses, pose))

        np.savetxt(join(query_data_folder, 'poses.csv'), poses, delimiter=',')
        np.savetxt(join(query_data_folder, 'num_inliers.csv'), len_best_inliers, delimiter=',', fmt='%i')
