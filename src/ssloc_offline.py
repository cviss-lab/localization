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
from datetime import datetime

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
        self.detector = 'SuperPoint'
        self.matcher = 'SuperGlue'
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

            if s < 0.1:
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
                matches, kp1, kp2 = self.feature_matching(des1, des2, self.detector, self.matcher, fname1_local, fname2_local,
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
            return

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

        return T_m1_c2

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

    def feature_matching(self, des1, des2, detector, matcher, fname1=None, fname2=None, model=None, img1=None, img2=None):

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

            img1 = cv2.resize(img1, (img1.shape[1] // 8 * 8, img1.shape[0] // 8 * 8))  # input size shuold be divisible by 8
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

            matches = [None]*int(mkpts0.shape[0])
            kp1 = [None]*int(mkpts0.shape[0])
            kp2 = [None]*int(mkpts0.shape[0])
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


if __name__ == '__main__':

    data_folder = "/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4"

    K1 = np.loadtxt(join(data_folder, 'K1.txt'))

    image_dir = join(data_folder, 'rgb')
    num1_images = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
    poses = np.loadtxt(join(data_folder, 'poses.csv'), delimiter=",")
    n = Node(debug=False, data_folder=data_folder, create_new_anchors=False)
    query_mode = True
    localisation_mode = True
    print("Test")
    if n.create_new_anchors:
        for i in range(num1_images):
            I1 = cv2.imread(join(data_folder, 'rgb', str(i + 1) + '.jpg'))
            D1 = cv2.imread(join(data_folder, 'depth', str(i + 1) + '.png'), cv2.IMREAD_UNCHANGED)
            pose1 = poses[i][1:8]
            n.K1 = K1
            n.counter = i
            n.create_anchor(I1, D1, pose1)
            print(i)


    if query_mode:
        query_data_folder = '/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2'
        query_img_dir = join(query_data_folder, 'images')
        sfm_file = join(query_data_folder, '0_6_55_img_ultra/reconstruction_global/sfm_data.json')


        T_m2_c2_dict = utils.return_T_M2_C2(sfm_file)
        q_list = []
        t_list = []
        for num, filename in enumerate(os.listdir(query_img_dir)):
            query_img_idx = int(os.path.splitext(filename)[0])
            image_path = os.path.join(query_img_dir, filename)
            print(query_img_idx)
            print(image_path)
            I2 = cv2.imread(image_path)
            K2 = np.loadtxt(join(data_folder, 'K2.txt'))
            T_m2_c2 = T_m2_c2_dict[query_img_idx]
            T_m1_c2 = n.callback_query(I2, K2)
            T_m1_m2 = T_m1_c2.dot(np.linalg.inv(T_m2_c2))
            R = Rotation.from_matrix(T_m1_m2[:3, :3])
            q = R.as_quat()
            q_list.append(q)
            t = T_m1_m2[:3, 3].T
            t_list.append(t)

        q_array = np.array(q_list)
        # np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/q_array.txt", q_array, delimiter=',')

        t_array = np.array(t_list)
        # np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/t_array.txt", t_array, delimiter=',')


    if localisation_mode:
        # q_array = np.loadtxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/q_array.txt", delimiter=',',
        #                      usecols=range(4))
        # t_array = np.loadtxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/t_array.txt", delimiter=',',
        #                      usecols=range(3))
        q_avg = np.average(q_array, axis=0)
        t_avg = np.average(t_array, axis=0)
        Rot_average = (Rotation.from_quat(q_avg).as_matrix())
        R_avg = Rot_average
        T_m1_m2_avg = np.eye(4)
        T_m1_m2_avg[:3, :3] = R_avg
        T_m1_m2_avg[:3, 3] = t_avg.reshape(-1)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/T_m1_m2"+dt_string+".txt",T_m1_m2_avg)

    print("TEST")
