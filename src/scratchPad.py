#!/usr/bin/env python3
import numpy as np
import sys
import os
from os.path import join, dirname, realpath
import cv2
import utils
from scipy.spatial.transform import Rotation
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import open3d as o3d
import copy
from datetime import datetime
import time
from operator import itemgetter

hloc_module = join(dirname(realpath(__file__)), 'hloc_toolbox')
sys.path.insert(0, hloc_module)

from hloc_toolbox import match_features, detect_features, search
from hloc_toolbox.hloc.matchers import loftr

torch.cuda.empty_cache()


class Node:

    def __init__(self, debug=False, data_folder=None, create_new_anchors=False):

        self.debug = debug
        self.map_frame_id = None
        self.robot_camera_frame_id = None
        self.send_unity_pose = False
        self.frame_rate = 1
        self.detector = "SuperPoint"
        self.matcher = "SuperGlue"
        self.sliding_average_buffer = 1
        self.results_dir = join(data_folder, 'results')
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

    def callback_query(self, *args):

        print('query image recieved!')
        I2 = args[0]
        K2 = args[1]

        print('Detecting local features in query image...')
        fname2_local = join(self.results_dir, 'local_features', 'local_q.h5')
        kp2, des2 = self.feature_detection(I2, self.detector, fname2_local, model=self.detector_model)

        print('Retrieving similar anchors...')
        fdir_db = join(self.results_dir, 'global_features')
        fname2_global = join(self.results_dir, 'global_q.h5')
        self.feature_detection(I2, 'netvlad', fname2_global, model=self.retrieval_model, id='q.jpg')

        pairs, scores = search.main(fname2_global, fdir_db, num_matches=20)

        print('\nsimilarity score between anchor and query: %.4f' % scores[0])
        pts2D_all = np.array([]).reshape(0, 2)
        pts3D_all = np.array([]).reshape(0, 3)

        matches1 = []
        ret_index1 = None

        for i, (p, s) in enumerate(zip(pairs, scores)):

            if i > 5:
                break  # terminate loop

            # if s < 0.1:  # originally 0.1
            #     continue  # skip the current iteration of the loop

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

            # img_matches = self.draw_matches_ros(I1, I2, kp1, kp2, matches)
            # img_matches_resize = utils.ResizeWithAspectRatio(img_matches, width=1920)
            # cv2.imshow('img', img_matches_resize)
            # cv2.waitKey(0)

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
            return np.zeros((4,4)), 0

        retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3D_all, pts2D_all, K2, None, flags=cv2.SOLVEPNP_P3P)

        # find relocalized pose of query image relative to robot camera
        R_ = cv2.Rodrigues(rvecs)[0]
        R = R_.T
        C = -R_.T.dot(tvecs)
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3, :3] = R
        T_m1_c2[:3, 3] = C.reshape(-1)

        # send localized pose relative to robot map
        # T_m1_c2 = self.send_reloc_pose(C, R, query_frame_id, timestamp_query)
        # if self.send_unity_pose:
        #     self.send_unity2map_pose(unity_pose, T_m1_c2, query_frame_id, timestamp_query)
        #
        # print('\nquery camera localized!\n')
        # #
        # # if not self.ros:
        # #     print('query camera transform:\n %s' % np.array2string(utils.Tmatrix_inverse(T_m1_c2)))
        #
        # # calculate errors from markers
        if self.debug:
            I1 = cv2.imread(join(self.results_dir, 'rgb', 'rgb_%i.png' % ret_index1))
            D1 = cv2.imread(join(self.results_dir, 'depth', 'depth_%i.png' % ret_index1), cv2.IMREAD_UNCHANGED)
            K1 = np.loadtxt(join(self.results_dir, 'K1.txt'))
            pose1 = np.loadtxt(join(self.results_dir, 'poses', 'pose_%i.txt' % ret_index1))
            T_c2_m1 = utils.Tmatrix_inverse(T_m1_c2)
            self.check_error(I1, I2, D1, pose1, K1, K2, kp1, kp2, matches1, 'interactive', T_c2_m1, inliers)

        return T_m1_c2, scores[0]

    def send_reloc_pose(self, C, R, query_frame_id, timestamp_query):
        R2 = np.eye(4)
        R2[:3, :3] = R
        q = tf.transformations.quaternion_from_matrix(R2)
        transform = utils.create_transform_stamped((C[0], C[1], C[2]),
                                                   (q[0], q[1], q[2], q[3]),
                                                   timestamp_query,
                                                   query_frame_id,
                                                   self.map_frame_id)
        # if self.ros:
        #     self.br.sendTransform(transform)
        T_m1_c2 = np.eye(4)
        T_m1_c2[:3, :3] = R
        T_m1_c2[:3, 3] = C.reshape(-1)
        self.T_m1_c2_buffer.append(T_m1_c2)
        if len(self.T_m2_c2_buffer) > self.sliding_average_buffer:
            self.T_m2_c2_buffer.pop(0)

        return T_m1_c2

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
            img1 = img1.astype(np.uint8)
            img2 = img2.astype(np.uint8)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

            img1 = cv2.resize(img1,
                              (img1.shape[1] // 8 * 8, img1.shape[0] // 8 * 8))  # input size shuold be divisible by 8
            img2 = cv2.resize(img2, (img2.shape[1] // 8 * 8, img2.shape[0] // 8 * 8))

            img1 = torch.from_numpy(img1)[None][None].cuda() / 255.
            img2 = torch.from_numpy(img2)[None][None].cuda() / 255.
            batch = {'image0': img1, 'image1': img2}

            matches = model(batch)

            """
            Need to go from matches which is a dictionary of torch tensors to opencv matches
            https://docs.opencv.org/3.4/d4/de0/classcv_1_1DMatch.html#ab9800c265dcb748a28aa1a2d4b45eea4
            """

            mkpts0 = matches['mkpts0_f'].cpu().numpy()
            mkpts1 = matches['mkpts1_f'].cpu().numpy()

            matches = [None] * int(mkpts0.shape[0])
            kp1 = [None] * int(mkpts0.shape[0])
            kp2 = [None] * int(mkpts0.shape[0])
            for i in range(mkpts0.shape[0]):
                matches[i] = cv2.DMatch(i, i, 0)
                kp1[i] = cv2.KeyPoint(mkpts0[i][0], mkpts0[i][1], 0, 1, -1)
                kp2[i] = cv2.KeyPoint(mkpts1[i][0], mkpts1[i][1], 0, 1, -1)

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

    def check_error(self, I1, I2, D1, pose1, K1, K2, kp1, kp2, matches, marker_type, T_c2_m1, inliers):

        # pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        # pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        # x,y,z = utils.project_2d_to_3d(pts1.T,K1,D1,h=5)
        # pts3D = np.array([x,y,z]).T

        # idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])

        # pts3D = pts3D[idx]
        # pts2D = pts2[idx]

        # retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D, pts2D, K2, None,flags=cv2.SOLVEPNP_P3P)

        # R = cv2.Rodrigues(rvecs)[0]
        # t = tvecs
        # T = np.hstack([R, t])
        T = T_c2_m1[:3, :]
        P2 = np.dot(K2, T)

        marker_length = 0.12

        fname1 = join(self.results_dir, '_reproj1.png')
        fname2 = join(self.results_dir, '_reproj2.png')
        fname_matches = join(self.results_dir, '_matches.png')

        res = self.reproj_error(P2, I1, D1, pose1, K1, I2, K2, marker_length, fname1, fname2, marker_type)

        if res is None:
            return
        elif marker_type == 'marker':
            error0, error1, error2, error3 = res
        elif marker_type == 'manual' or marker_type == 'interactive':
            error0 = res
            error1 = np.nan
            error2 = np.nan
            error3 = np.nan

        n_inliers = len(inliers)

        print('Reprojection Error: %.3f' % error0)
        print('Positional Error (H): %.3f' % error1)
        print('Positional Error (in-plane): %.3f' % error2)
        print('Positional Error (out-of-plane): %.3f' % error3)
        print('Number of inlier matches: %i' % n_inliers)

        self.draw_matches(I1, I2, kp1, kp2, matches, fname_matches)

    def reproj_error(self, P2, I1, D1, pose1, K1, I2, K2, marker_length, fname1, fname2, marker_type, PLOT_FIGS=False):

        if marker_type == 'manual':
            m = np.array([0.55, 0.65])
            s = 'test'
            m1 = utils.detect_manual(I1, m, s, h=0.015)
            m2 = utils.detect_manual(I2, m, s, h=0.015)
            if m1 is None or m2 is None or len(m1) == 0 or len(m2) == 0:
                return

            id1 = m1[0]['id']
            id2 = m2[0]['id']

            m1 = m1[0]['coords']
            m2 = m2[0]['coords']
        elif marker_type == 'interactive':
            m1 = self.interactive_bbox(I1)
            m2 = self.interactive_bbox(I2)
        elif marker_type == 'marker':
            m1 = utils.detect_markers(I1, xy_array=True)
            m2 = utils.detect_markers(I2, xy_array=True)
            if m1 is None or m2 is None or len(m1) == 0 or len(m2) == 0:
                return

            id1 = m1[0]['id']
            id2 = m2[0]['id']

            m1 = m1[0]['coords']
            m2 = m2[0]['coords']

        x, y, z = utils.project_2d_to_3d(m1, K1, D1, h=5)
        m_3D_c = np.array([x, y, z, np.ones(x.shape[0])])

        tc = pose1[:3]
        qc = pose1[3:]
        T_m1_c1 = tf.transformations.quaternion_matrix(qc)
        T_m1_c1[:3, 3] = tc

        m_3D = T_m1_c1.dot(m_3D_c)
        m_3D = m_3D[:3, :] / m_3D[3, :]
        # m_3D = np.array([x,y,z])

        m2_proj = utils.ProjectToImage(P2, m_3D)
        reproj_error = np.sum(np.sqrt((m2 - m2_proj)[0, :] ** 2 + (m2 - m2_proj)[1, :] ** 2))

        x1, y1 = np.int32(m1[:, 0])
        x2, y2 = np.int32(m1[:, 1])
        x3, y3 = np.int32(m1[:, 2])
        x4, y4 = np.int32(m1[:, 3])

        import copy
        img = copy.copy(I1)
        t = int(np.max([1, img.shape[1] / 1000]))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=t)
        cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=t)
        cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), thickness=t)
        cv2.line(img, (x4, y4), (x1, y1), (0, 255, 0), thickness=t)

        cv2.imwrite(fname1, img)

        x1, y1 = np.int32(m2_proj[:, 0])
        x2, y2 = np.int32(m2_proj[:, 1])
        x3, y3 = np.int32(m2_proj[:, 2])
        x4, y4 = np.int32(m2_proj[:, 3])

        img = copy.copy(I2)
        t = int(np.max([1, img.shape[1] / 1000]))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=t)
        cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), thickness=t)
        cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), thickness=t)
        cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), thickness=t)

        cv2.imwrite(fname2, img)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('', cv2.resize(img, [int(img.shape[1] / 4), int(img.shape[0] / 4)]));
        cv2.waitKey(0)

        if PLOT_FIGS:
            plt.imshow(cv2.cvtColor(I2, cv2.COLOR_RGB2BGR)), plt.show()

        if marker_type == 'manual' or marker_type == 'interactive':
            return reproj_error
        elif marker_type == 'marker':
            m2_scaled = np.array([[-marker_length / 2, -marker_length / 2], [marker_length / 2, -marker_length / 2],
                                  [marker_length / 2, marker_length / 2], [-marker_length / 2, marker_length / 2]])
            H = cv2.findHomography(m2.T, m2_scaled)[0]

            m2_proj_scaled = cv2.perspectiveTransform(m2_proj.T.reshape(-1, 1, 2), H).reshape(-1, 2)

            diff = (m2_proj_scaled - m2_scaled).T
            reproj_error_scaled = np.mean(np.sqrt(diff[0, :] ** 2 + diff[1, :] ** 2))

            rvecs2, tvecs2, objpoints_m_check = cv2.aruco.estimatePoseSingleMarkers([m2.T], marker_length, K2,
                                                                                    (0, 0, 0, 0))

            objpoints_c1 = m_3D
            T_c2_c1 = np.vstack([T, [0, 0, 0, 1]])
            objpoints_m_check = objpoints_m_check.reshape(4, 3)
            R_c2_m = cv2.Rodrigues(rvecs2)[0]
            t_c2_m = tvecs2.reshape(3, 1)
            T_m_c2 = np.vstack([np.hstack([R_c2_m.T, -R_c2_m.T.dot(t_c2_m)]), [0, 0, 0, 1]])
            _objpoints_c1 = np.vstack([objpoints_c1.T, np.ones((1, 4))])
            _objpoints_c2 = np.dot(T_c2_c1, _objpoints_c1)
            _objpoints_m = np.dot(T_m_c2, _objpoints_c2)
            objpoints_m = (_objpoints_m[:3, :] / _objpoints_m[3, :]).T
            diff = objpoints_m.mean(axis=0)
            error2 = np.linalg.norm(diff[:2])
            error3 = np.linalg.norm(diff[2])

            if id1 != id2:
                for _ in range(4):
                    m2_proj_scaled = np.array([m2_proj_scaled[i, :] for i in [1, 2, 3, 0]])
                    m2_proj = np.array([m2_proj[:, i] for i in [1, 2, 3, 0]]).T
                    diff = (m2_proj_scaled - m2_scaled).T
                    reproj_error_scaled2 = np.mean(np.sqrt(diff[0, :] ** 2 + diff[1, :] ** 2))
                    reproj_error2 = np.sum(np.sqrt((m2 - m2_proj)[0, :] ** 2 + (m2 - m2_proj)[1, :] ** 2))
                    if reproj_error_scaled2 < reproj_error_scaled:
                        reproj_error_scaled = reproj_error_scaled2
                        reproj_error = reproj_error2

            return reproj_error, reproj_error_scaled, error2, error3

    def draw_matches(self, I1, I2, kp1, kp2, matches, fname, PLOT_FIGS=False):

        img = cv2.drawMatches(I1, kp1, I2, kp2, matches, None, flags=2)
        if PLOT_FIGS:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), plt.show()

        cv2.imwrite(fname, img)

    def draw_matches_ros(self, I1, I2, kp1, kp2, matches):
        img = cv2.drawMatches(I1, kp1, I2, kp2, matches, None, flags=2)
        return img

    def interactive_bbox(self, image):

        x_coord = []
        y_coord = []

        def interactive_win(event, u, v, flags, param):

            t = int(np.max([1, image2.shape[1] / 1000]))

            if event == cv2.EVENT_LBUTTONDOWN:
                x_coord.append(u)
                y_coord.append(v)
                if len(x_coord) >= 2:
                    cv2.line(image2, (x_coord[-1], y_coord[-1]), (x_coord[-2], y_coord[-2]), (0, 255, 0), thickness=t)
                if len(x_coord) == 4:
                    cv2.line(image2, (x_coord[-1], y_coord[-1]), (x_coord[0], y_coord[0]), (0, 255, 0), thickness=t)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', interactive_win)

        image2 = image

        while (1):
            cv2.imshow('image', image2)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or k == 13:  # 'Esc' or 'Enter' Key
                break

        x_coord = np.array(x_coord).reshape(1, -1)
        y_coord = np.array(y_coord).reshape(1, -1)

        return np.vstack([x_coord, y_coord])

    def create_offline_anchors(self):
        image_dir = join(data_folder, 'rgb')
        num_images = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        poses = np.loadtxt(join(data_folder, 'poses.csv'), delimiter=",")
        K1 = np.loadtxt(join(data_folder, 'K1.txt'))

        for i in range(num_images):
            torch.cuda.empty_cache()
            I1 = cv2.imread(join(data_folder, 'rgb', str(i + 1) + '.jpg'))
            D1 = cv2.imread(join(data_folder, 'depth', str(i + 1) + '.png'), cv2.IMREAD_UNCHANGED)
            pose1 = poses[i][1:8]
            n.K1 = K1
            n.counter = i
            n.create_anchor(I1, D1, pose1)


def perform_querying(map_folder, query_folder):
    query_img_dir = join(query_folder, 'images')
    # sfm_file = join(query_folder, 'known_ultra_unblurred/reconstruction_global/sfm_data.json')
    sfm_file = join(query_folder, 'unknown_ultra_unblurred/reconstruction_global/sfm_data.json')

    T_m2_c2_dict = utils.return_T_M2_C2(sfm_file)

    q_list = []
    t_list = []
    q_dict = {}
    t_dict = {}
    T_m1_c2_dict = {}
    score_dict = {}

    for num, filename in enumerate(os.listdir(query_img_dir)):
        if num < 200:
            query_img_idx = int(os.path.splitext(filename)[0])
            image_path = os.path.join(query_img_dir, filename)
            print(query_img_idx)
            print(image_path)
            I2 = cv2.imread(image_path)
            K2 = np.loadtxt(join(data_folder, 'K2.txt'))
            T_m2_c2 = T_m2_c2_dict[query_img_idx]

            T_m1_c2, score = n.callback_query(I2, K2)

            if T_m1_c2 is not None:
                T_m1_c2_dict[query_img_idx] = T_m1_c2
                T_m1_m2 = T_m1_c2.dot(np.linalg.inv(T_m2_c2))
                R = Rotation.from_matrix(T_m1_m2[:3, :3])
                q = R.as_quat()
                t = T_m1_m2[:3, 3].T
                q_dict[query_img_idx] = q
                t_dict[query_img_idx] = t
                score_dict[query_img_idx] = score
                # q_list.append(q)
                # t_list.append(t)

    with open(join(query_folder, "T_m1_c2_dict.pkl"), 'wb') as f:
        pickle.dump(T_m1_c2_dict, f)
    with open(join(query_folder, "T_m2_c2_dict.pkl"), 'wb') as f:
        pickle.dump(T_m2_c2_dict, f)
    # with open(join(query_folder, "q_dict.pkl"), 'wb') as f:
    #     pickle.dump(q_dict, f)
    # with open(join(query_folder, "t_dict.pkl"), 'wb') as f:
    #     pickle.dump(t_dict, f)
    with open(join(query_folder, "score_dict.pkl"), 'wb') as f:
        pickle.dump(score_dict, f)

    # return q_dict, t_dict, T_m1_c2_dict, T_m2_c2_dict, score_dict
    return T_m1_c2_dict, T_m2_c2_dict, score_dict

def perform_scaling(t_m1_c2, t_m2_c2,best_queries):
    dict_filter = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

    top_t_m1_c2_dict = dict_filter(t_m1_c2, best_queries)
    top_t_m2_c2_dict = dict_filter(t_m2_c2, best_queries)
    centers_m1 = []
    centers_m2 = []
    # for query in sorted_t_m1_c2_dict:
    for query in top_t_m1_c2_dict:
        C_m1 = (top_t_m1_c2_dict[query][:3, 3]).reshape(-1, 1)
        C_m2 = (top_t_m2_c2_dict[query][:3, 3]).reshape(-1, 1)
        centers_m1.append(C_m1)
        centers_m2.append(C_m2)

    pts1 = np.array(centers_m1).reshape(-1, 3)
    pts2 = np.array(centers_m2).reshape(-1, 3)

    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    z1 = pts1[:, 2]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    z2 = pts2[:, 2]
    # Create Figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x1, y1, z1, marker='o', s=20, label='m1')
    ax.scatter3D(x2, y2, z2, marker='o', s=20, label='m2')
    utils.set_axes_equal(ax)
    ax.legend(loc=1)
    plt.show()
    dist1 = np.linalg.norm(pts1 - pts1[:, None], axis=-1)
    dist2 = np.linalg.norm(pts2 - pts2[:, None], axis=-1)

    scale = np.divide(dist1, dist2)
    top_scale = scale[0, 1]

    scaled_t_m2_c2_dict = t_m2_c2.copy()
    scaled_centers = {}
    for query in scaled_t_m2_c2_dict:
        print(t_m2_c2[query])

        scaled_centers = t_m2_c2[query][:3, 3] * top_scale
        scaled_t_m2_c2_dict[query][:3, 3] = scaled_centers
        # if query == 99:

        print(scaled_t_m2_c2_dict[query])
    print(t_m2_c2[query])
    print(scaled_t_m2_c2_dict[query])
    return scaled_t_m2_c2_dict, top_scale


# def perform_map_localisation(quat, translations):
#     q_array = np.array(list(quat.values()))
#     t_array = np.array(list(translations.values()))
#
#     q_avg = np.average(q_array, axis=0)
#     t_avg = np.average(t_array, axis=0)
#     Rot_average = (Rotation.from_quat(q_avg).as_matrix())
#     R_avg = Rot_average
#     T_m1_m2_avg = np.eye(4)
#     T_m1_m2_avg[:3, :3] = R_avg
#     T_m1_m2_avg[:3, 3] = t_avg.reshape(-1)
#
#     return T_m1_m2_avg

def perform_map_localisation(t_m1_c2, t_m2_c2, query_folder, queries):
    query_img_dir = join(query_folder, 'images')
    q_dict = {}
    t_dict = {}
    for query in queries:
        join(query_img_dir,str(181)+".jpg")
        T_m1_c2 = t_m1_c2[query]
        T_m2_c2 = t_m2_c2[query]
        T_m1_m2 = T_m1_c2.dot(np.linalg.inv(T_m2_c2))
        R = Rotation.from_matrix(T_m1_m2[:3, :3])
        q = R.as_quat()
        t = T_m1_m2[:3, 3].T
        q_dict[query] = q
        t_dict[query] = t
    # for num, filename in enumerate(os.listdir(query_img_dir)):
    #     if query in queries:
    #         query_img_idx = int(os.path.splitext(filename)[0])
    #         # if T_m1_c2 is not None:
    #         T_m1_c2 = t_m1_c2[query_img_idx]
    #         T_m2_c2 = t_m2_c2[query_img_idx]
    #         T_m1_m2 = T_m1_c2.dot(np.linalg.inv(T_m2_c2))
    #         R = Rotation.from_matrix(T_m1_m2[:3, :3])
    #         q = R.as_quat()
    #         t = T_m1_m2[:3, 3].T
    #         q_dict[query_img_idx] = q
    #         t_dict[query_img_idx] = t

    q_array = np.array(list(q_dict.values()))
    t_array  = np.array(list(t_dict.values()))

    q_avg = np.average(q_array, axis=0)
    t_avg = np.average(t_array, axis=0)
    Rot_average = (Rotation.from_quat(q_avg).as_matrix())
    R_avg = Rot_average
    T_m1_m2_avg = np.eye(4)
    T_m1_m2_avg[:3, :3] = R_avg
    T_m1_m2_avg[:3, 3] = t_avg.reshape(-1)

    return T_m1_m2_avg

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def colored_ICP(src, trgt):
    # colored pointcloud registration This is implementation of following paper: J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    print("3. Colored point cloud registration")
    start = time.time()
    current_transformation = np.identity(4)

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        src_down = src.voxel_down_sample(radius)
        trgt_down = trgt.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        trgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        icp_result = o3d.pipelines.registration.registration_colored_icp(
            src_down, trgt_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = icp_result.transformation
        print(icp_result)
    print("Colored ICP took %.3f sec.\n" % (time.time() - start))
    draw_registration_result_original_color(src, trgt, icp_result.transformation)
    return icp_result


if __name__ == '__main__':

    data_folder = "/home/jp/Desktop/Rishabh/Handheld/22-08-08-StructuresLab3dSpalling-processed"
    query_data_folder = '/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2/08_08_2022'

    # n = Node(debug=False, data_folder=data_folder, create_new_anchors=False)
    #
    query_mode = False
    map_localisation_mode = True
    map_registration_mode = False
    icp_test = True

    scaling = True
    # if n.create_new_anchors:
    #     n.create_offline_anchors()

    if query_mode:
        T_m1_c2, T_m2_c2, scores = perform_querying(data_folder, query_data_folder)
    else:
        with open(join(query_data_folder, "T_m1_c2_dict.pkl"), 'rb') as f:
            T_m1_c2 = pickle.load(f)
        with open(join(query_data_folder, "T_m2_c2_dict.pkl"), 'rb') as f:
            T_m2_c2 = pickle.load(f)
        with open(join(query_data_folder, "score_dict.pkl"), 'rb') as f:
            scores = pickle.load(f)

    sorted_scores = dict(sorted(scores.items(), key=itemgetter(1)))
    top_n = 10
    top_queries = (list(sorted_scores.keys())[-top_n:])

    if scaling:
        T_m2_c2_dict, scale_num = perform_scaling(T_m1_c2, T_m2_c2, top_queries)


    sorted_T_m1_c2_dict = dict(sorted(T_m1_c2.items(), key=itemgetter(0)))
    sorted_T_m2_c2_dict = dict(sorted(T_m2_c2.items(), key=itemgetter(0)))

    dict_filter = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])

    top_t_m1_c2_dict = dict_filter(sorted_T_m1_c2_dict, top_queries)
    top_t_m2_c2_dict = dict_filter(sorted_T_m2_c2_dict, top_queries)


    if map_localisation_mode:
        # T_m1_m2_avg = perform_map_localisation(q, t)
        T_m1_m2_avg = perform_map_localisation(top_t_m1_c2_dict, top_t_m2_c2_dict,query_data_folder, top_queries)

        np.savetxt(join(query_data_folder, "T_m1_m2_top_%d_avg.txt" % top_n), T_m1_m2_avg)
        # np.savetxt(join(query_data_folder, "T_m1_m2_top_avg.txt"), T_m1_m2_avg)


    if map_registration_mode:
        print("1. Load two point clouds and show initial pose")

        # Load the source map (smaller map)
        source = o3d.io.read_point_cloud(
            "/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2/08_08_2022/unknown_ultra_unblurred/reconstruction_global/scaled_transformed_MVG_colorized.ply")

        # Load the target map (larger map)
        target = o3d.io.read_point_cloud(
            "/home/jp/Desktop/Rishabh/Handheld/22-08-08-StructuresLab3dSpalling-processed/r3live_output/rgb_pt.pcd")
        # Add the initial transformation (if available, otherwise Identity matrix)

        # scaled_source = source.scale(scale_num, source.get_center())

        # trans_init = T_m1_m2_avg
        trans_init = np.eye(4)

        source.transform(trans_init)
        source.estimate_normals()
        target.estimate_normals()
        # draw_registration_result_original_color(source, target, current_transformation)
        # draw_registration_result_original_color(source, target)
        draw_registration_result_original_color(source, target, np.eye(4))
        result_icp = colored_ICP(source, target)
        # Saves the 4x4 transform as a txt file
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

        np.savetxt(join(data_folder, "unblurred_T_colored_icp_" + dt_string + ".txt"), result_icp.transformation)
        np.savetxt(join(data_folder, "unblurred_T_colored_icp_total_" + dt_string + ".txt"),
                   np.dot(result_icp.transformation, trans_init))

        print(result_icp.transformation)


    if icp_test:
        threshold = 1
        # Load the source map (smaller map)
        source = o3d.io.read_point_cloud(
            "/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2/08_08_2022/conference_outputs/scaled_transformed_MVG_colorized.ply")

        # Load the target map (larger map)
        target = o3d.io.read_point_cloud(
            "/home/jp/Desktop/Rishabh/Handheld/22-08-08-StructuresLab3dSpalling-processed/r3live_output/rgb_pt.pcd")
        # Add the initial transformation (if available, otherwise Identity matrix)

        trans_init = np.eye(4)

        print("Initial alignment")
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, threshold, trans_init)
        print(evaluation)
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result_original_color(source, target, reg_p2p.transformation)

        now = datetime.now()
        # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        # np.savetxt(join("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2/08_08_2022",
        #                 "conference_icp_Transform" + dt_string + ".txt"), reg_p2p.transformation)

    # sorted_scored = dict(sorted(score_dict.items(), key=itemgetter(1)))
    # top_retrievals = list(dict(sorted(score_dict.items(), key=itemgetter(1))).keys())[-1], \
    #                  list(dict(sorted(score_dict.items(), key=itemgetter(1))).keys())[-2]

    # q_array = np.vstack((q_dict[top_retrievals[0]], q_dict[top_retrievals[1]]))
    # t_array = np.vstack((t_dict[top_retrievals[0]], t_dict[top_retrievals[1]]))
    # q_avg = np.average(q_array, axis=0)
    # t_avg = np.average(t_array, axis=0)
    # Rot_average = (Rotation.from_quat(q_avg).as_matrix())
    # R_avg = Rot_average
    # T_m1_m2_avg = np.eye(4)
    # T_m1_m2_avg[:3, :3] = R_avg
    # T_m1_m2_avg[:3, 3] = t_avg.reshape(-1)
    # np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/optimised_T_m1_m2.txt",
    #            T_m1_m2_avg)
    #
    # centers_m1 = []
    # centers_m2 = []
    # C_m1 = np.vstack(((sorted_T_m1_c2_dict[top_retrievals[0]][:3, 3]).reshape(-1, 1),(sorted_T_m1_c2_dict[top_retrievals[1]][:3, 3]).reshape(-1, 1)))
    # C_m2 =  np.vstack(((sorted_T_m2_c2_dict[top_retrievals[0]][:3, 3]).reshape(-1, 1),(sorted_T_m2_c2_dict[top_retrievals[1]][:3, 3]).reshape(-1, 1)))
    # centers_m1.append(C_m1)
    # centers_m2.append(C_m2)
    #
    # pts1 = np.array(centers_m1).reshape(-1, 3)
    # pts2 = np.array(centers_m2).reshape(-1, 3)
    # dist1 = np.linalg.norm(pts1 - pts1[:, None], axis=-1)
    # dist2 = np.linalg.norm(pts2 - pts2[:, None], axis=-1)
    #
    # scale = np.divide(dist1, dist2)

    #
    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    # np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/T_m1_m2_" + dt_string + ".txt",
    #            T_m1_m2_avg)
    print(T_m1_m2_avg)
    pts2_transformed = T_m1_m2_avg.dot(np.hstack((pts2, np.ones(top_n).reshape(top_n, 1))).T).T[:, :3]
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    z1 = pts1[:, 2]
    x2 = pts2_transformed[:, 0]
    y2 = pts2_transformed[:, 1]
    z2 = pts2_transformed[:, 2]
    # Create Figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x1, y1, z1, marker='o', s=20, label='m1')
    ax.scatter3D(x2, y2, z2, marker='o', s=20, label='m2')
    utils.set_axes_equal(ax)
    ax.legend(loc=1)
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # p_i = np.vstack((pts1[0, :], pts2_transformed[0, :]))
    # x_i = p_i[:, 0]
    # y_i = p_i[:, 1]
    # z_i = p_i[:, 2]
    # ax.scatter3D(x_i, y_i, z_i, marker='o', s=20)
    # p_i = np.vstack((pts1[5, :], pts2_transformed[5, :]))
    # x_i = p_i[:, 0]
    # y_i = p_i[:, 1]
    # z_i = p_i[:, 2]
    # ax.scatter3D(x_i, y_i, z_i, marker='o', s=20)
    #
    # utils.set_axes_equal(ax)
    # plt.show()

    print("Test")
