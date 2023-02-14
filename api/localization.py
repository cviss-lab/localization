import numpy as np
import os
from os.path import join, dirname, realpath
import cv2
import torch
import json
from pathlib import Path
import uuid
import shutil

from hloc.matchers import loftr
from hloc import extract_features, match_features

from libs.utils.projection import *
from libs.utils.loader import *
from hloc import wrapper

torch.cuda.empty_cache()

class Localizer:

    def __init__(self, loader: LocalLoader, detector: str ='loftr', matcher: str ='loftr', retrieval: str ='netvlad'):

        self.loader = loader

        (
            self.camera_matrix,
            self.image_width,
            self.image_height,
        ) = loader.load_intrinsics()
        self.poses = loader.load_poses()
        self.data_dir = loader.working_dir

        self.detector = detector
        self.matcher = matcher
        self.retrieval = retrieval

        self.matcher_model = None
        self.detector_model = None
        self.retrieval_model = None

        self.retrieval_path = wrapper.get_features_path(os.path.join(self.data_dir,'features'), self.retrieval)
        self.feature_path = None

        if not os.path.exists(self.retrieval_path):
            self.build_database()        

        self.rgb_dict = loader.load_imgs_dict(self.poses, "rgb")
        self.depth_dict = loader.load_imgs_dict(self.poses, "depth")        
        self.poses = loader.load_poses()

        self.load_models()

    def load_depth(self, img_idx):
        return self.loader.load_depth(os.path.join("depth", self.depth_dict[img_idx]))

    def load_rgb(self, img_idx):
        return self.loader.load_depth(os.path.join("rgb", self.rgb_dict[img_idx]))        

    def get_pose(self, img_idx):
        pose = np.array(self.poses[img_idx])
        return pose

    def build_database(self):

        data = Path(self.data_dir)

        images = data / 'rgb'
        features_dir = data / 'features'

        retrieval_conf = extract_features.confs['netvlad']
        self.retrieval_path = extract_features.main(retrieval_conf, images, features_dir)

        if self.detector != 'loftr' and self.detector_model != None:
            feature_conf = extract_features.confs[self.detector]
            self.feature_path = extract_features.main(feature_conf, images, features_dir)

        wrapper.unwrap_features_db(self.feature_path, self.data_dir)
            

    def callback_query(self, I2, K2, max_reproj_error=8, retrieved_anchors=5, similarity_threshold=0.1):

        print('query image recieved!')

        tmp_dir = join('/tmp/features',str(uuid.uuid4())) 
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)

        print('Detecting local features in query image...')
        kp2, des2 = self.feature_detection(I2, fname='local_q.h5',
                                           img_name='q.jpg',dataset=tmp_dir)

        print('Retrieving similar anchors...')
        
        pairs, scores = self.search(I2, img_name='q.jpg',dataset=tmp_dir)

        print('\nsimilarity score between anchor and query: %.4f' % scores[0])
        pts2D_all = np.array([]).reshape(0, 2)
        pts3D_all = np.array([]).reshape(0, 3)

        for i, (p, s) in enumerate(zip(pairs, scores)):

            if i > retrieved_anchors:
                break  # terminate loop

            if s < similarity_threshold:
                continue  # skip the current iteration of the loop

            if '.jpg' in p[1]:
                ret_index = int(p[1].replace('.jpg', ''))
            elif '.png' in p[1]:
                ret_index = int(p[1].replace('.png', ''))
            else:
                raise Exception("Invalid image extension")
                
            print('retrieved anchor %i\n' % ret_index)

            # database images
            I1 = self.load_rgb(ret_index)
            D1 = self.load_depth(ret_index)
            K1 = self.camera_matrix
            pose1 = self.get_pose(ret_index)

            if self.detector == "loftr":
                des1 = None
                des2 = None
                fname1_local = None
                fname2_local = None
            elif self.detector_model is not None:
                fname1_local = join(self.data_dir, 'features', 'local_%i.h5' % ret_index)
                kp1 = wrapper.load_features(fname1_local)
                des1 = None  # not used for superpoint
            else:
                kp1, des1 = self.feature_detection(I1)

            print('Matching features in query image...')
            res = self.feature_matching(des1, des2, img1=I1, img2=I2)

            if self.matcher == "loftr":
                matches, kp1, kp2 = res
            else:
                matches, _, _ = res

            if len(matches) > 10:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                pts3D, valid_ind = project_2d_to_3d(pts1.T, D1, K1, pose1,self.image_width, self.image_height, return_valid_ind=True)                
                pts3D = pts3D.T

                # remove points with invalid depth
                pts2D = pts2[valid_ind]

                pts2D_all = np.vstack([pts2D_all, pts2D])
                pts3D_all = np.vstack([pts3D_all, pts3D])

            # self.draw_matches(I1, I2, kp1, kp2, matches)                  

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
        print('\nInliers in View 1: {:d}' .format(num_inliers))

        # delete temp matches dir
        shutil.rmtree(tmp_dir, ignore_errors=True) 

        return T_m1_c2, num_inliers

    def feature_detection(self, I, fname=None, img_name=None, dataset=None):

        if self.detector == 'SIFT':
            gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
        elif self.detector == 'ORB':
            orb = cv2.ORB_create(nfeatures=5000)
            kp, des = orb.detectAndCompute(I, None)
        elif self.detector == 'SURF':
            surf = cv2.xfeatures2d.SURF_create()
            kp, des = surf.detectAndCompute(I, None)
        elif self.detector == "loftr":
            return None, None
        else:
            kp = wrapper.detect(I, fname, self.detector, model=self.detector_model, img_name=img_name, dataset=dataset)
            des = None

        return kp, des

    def search(self, I, fname=None, img_name=None, num_matches=5, dataset=None):
        if fname is None:
            fname= wrapper.get_features_path(dataset, self.retrieval)
        wrapper.detect(I, fname, self.retrieval, model=self.retrieval_model, img_name=img_name, dataset=dataset)
        pairs, scores = wrapper.search(fname, self.retrieval_path, num_matches=num_matches, dataset=dataset)   
        return pairs, scores     

    def feature_matching(self, des1, des2, img1=None,
                         img2=None):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            if self.matcher == 'BF':
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            elif self.matcher == 'FLANN':
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
            else:
                raise ValueError('Invalid matcher: {}'.format(self.matcher))
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            matches = good
            kp1 = None
            kp2 = None
        elif self.matcher == "loftr":

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

            matches = self.matcher_model(batch)

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
        # else:
        #     matches = wrapper.match(fname1, fname2, self.matcher, model=self.matcher_model)

        return matches

    def load_models(self):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            return
        else:   
            if self.detector != "loftr" and self.matcher != "loftr":
                self.detector_model = self.load_detector(self.detector)
            
            self.matcher_model = self.load_matcher(self.matcher)

        if self.retrieval == 'netvlad':
            self.retrieval_model = self.load_detector('netvlad')
            print('Loaded netvlad model')

    def load_detector(self, detector):
        feature_conf = extract_features.confs[detector]
        model = extract_features.load_model(feature_conf)
        return model        

    def load_matcher(self, matcher):
        if matcher == 'loftr':
            default_conf = {
                'weights': 'outdoor_ds.ckpt',
                'max_num_matches': 5000,
            }
            model = loftr.loftr(default_conf)
            print('Loaded LoFTR model')
        else:
            match_conf = match_features.confs[matcher]
            model = match_features.load_model(match_conf)
        return model  

    def draw_matches(self, I1, I2, kp1, kp2, matches):
        import matplotlib.pyplot as plt
        img = cv2.drawMatches(I1, kp1, I2, kp2, matches, None, flags=2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), plt.show()        

if __name__ == '__main__':
    data_folder = '/home/zaid/datasets/test_data/sample_data3'
    localizer = Localizer(LocalLoader(data_folder))
    # localizer.build_database()

    I2 = cv2.imread('/home/zaid/datasets/rgb_47.png')
    K2 = np.array([[6.380850830078125000e+02, 0.000000000000000000e+00, 6.356729736328125000e+02],
                   [0.000000000000000000e+00, 6.363466796875000000e+02, 3.661477050781250000e+02],
                   [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    T_m1_c2, len_best_inliers = localizer.callback_query(I2,K2)