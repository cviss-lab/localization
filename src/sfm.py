from copyreg import pickle
from bundle_adjustment import BundleAdjustment2
import sys
import os
from os.path import join, dirname, realpath
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from extrinsic2pyramid import CameraPoseVisualizer
import pickle
import time

hloc_module=join(dirname(realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import match_features, detect_features, search

class FeaturePoint():
    def __init__(self, u, v, img_id, feature_id):
        self.u = u
        self.v = v
        self.img_id = img_id
        self.feature_id = feature_id

        self.img_id_repeated = []
        self.u_repeat = []
        self.v_repeat = []

    def set_3d(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

    def repeated_in(self, img_id, u, v):
        self.img_id_repeated.append(img_id)
        self.u_repeat.append(u)
        self.v_repeat.append(v)

class TrackedFeaturePoints():
    def __init__(self):
        self.feature_points = []

    def to_numpy_2D(self,img_id=None):
        pts = np.zeros((len(self.feature_points),2),dtype=np.float32)
        for i in range(len(self.feature_points)):
            if self.feature_points[i].img_id == img_id or img_id is None:
                pts[i,0] = self.feature_points[i].u
                pts[i,1] = self.feature_points[i].v
        return pts

    def add(self, feature_point):
        self.feature_points.append(feature_point)

    def add_set(self, img_id, pts, tol=0.01):
        if len(self.feature_points) == 0:
            ind_tracked2 = []
        else:
            pts_stored = self.to_numpy_2D(img_id) # get the tracked points from the previous image
            ind_tracked1, ind_tracked2 = self.tracker(pts_stored,pts,tol=tol)

        for i in range(pts.shape[0]):
            if i in ind_tracked2:
                i2 = np.where(ind_tracked2==i)[0][0]
                self.feature_points[i2].repeated_in(img_id,pts[i,0],pts[i,1])
            else:
                feature_point = FeaturePoint(pts[i,0],pts[i,1],img_id)
                self.add(feature_point)

    def tracker(self,pts01,pts02,tol=0.01):

        A = np.sqrt(pts01.dot(pts02.T))
        d = np.linalg.norm(pts02,axis=1).T
        e = np.absolute(A-d)

        tracked = e < tol
        ind_tracked2 = np.argmax(tracked==1,axis=1)
        ind_tracked2 = ind_tracked2[np.max(tracked==1,axis=1)]
        
        ind_tracked1 = np.arange(pts01.shape[0])
        ind_tracked1 = ind_tracked1[np.max(tracked==1,axis=1)]

        er = pts01[ind_tracked1] - pts02[ind_tracked2]
        er = np.linalg.norm(er,axis=1)
        ind_tracked1 = ind_tracked1[er<tol]
        ind_tracked2 = ind_tracked2[er<tol]

        return ind_tracked1,ind_tracked2    

class SfM():
    def __init__(self,data_folder,create_new_anchors=True):
        self.counter = 0
        self.FOV = 120
        self.pts_idx = []
        self.detector = 'SuperPoint'
        self.matcher = 'SuperGlue'
        self.data_folder = data_folder   
        self.results_dir = join(data_folder,'results')
        self.create_new_anchors = create_new_anchors
        self.tracked_features = TrackedFeaturePoints()

        self.image_dir = join(self.data_folder, 'rgb')
        self.num_images = len([name for name in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, name))]) 
        # self.num_images = 3

        self.poses = np.zeros((self.num_images,7))
        self.poses[:,6] = 1  
        self.points_3d = np.empty((0,3))

        # self.init_viz(0,0,0)        

        utils.make_dir(join(self.results_dir), delete_if_exists=self.create_new_anchors)
        utils.make_dir(join(self.results_dir,'matches'))
        utils.make_dir(join(self.results_dir,'local_features'))
        utils.make_dir(join(self.results_dir,'global_features'))

    def slam(self):

        if self.create_new_anchors:
            self.load_models()
            kp0 = None; des0 = None; f0 = None
            pts0_l = []
            pts1_l = []
            matches_l = []

            # feauture detection and sequential matching                
            for i in range(self.num_images):
                img_panorama = cv2.imread(join(self.image_dir, '%i.jpg' % (i+1)))
                I_l, _, K = utils.fun_rectify_views(img_panorama, self.FOV)
                img = I_l[0]

                # process first frame
                if kp0 is None:
                    kp0, des0, f0 = self.create_anchor(img)
                    continue
                # process other frames
                kp1, des1, f1 = self.create_anchor(img)

                matches,pts0,pts1 = self.match_anchors(des0,des1,kp0,kp1,f0,f1)
                pts0_l.append(pts0)
                pts1_l.append(pts1) 
                matches_l.append(matches) 

                kp0 = kp1
                des0 = des1
                f0 = f1

                print('\nProcessed image %i/%i\n' % (i,self.num_images))

            self.save_matches(pts0_l,pts1_l,matches_l)
            np.savetxt(join(self.results_dir,'K.txt'),K)
        else:
            pts0_l,pts1_l,matches_l = self.load_matches()
            K = np.loadtxt(join(self.results_dir,'K.txt'))

        self.points_2d = []
        self.pts_idx = []

        # visual odometry
        for i in range(1,self.num_images):
            pts0 = pts0_l[i-1]
            pts1 = pts1_l[i-1]

            # self.tracked_features.add_set(i-1,pts0)
            # self.tracked_features.add_set(i,pts1)

            # pts1_from_next = pts0_l[i]
            # ind_tracked1,_ = self.tracker(pts1,pts1_from_next)
            # pts0 = pts0[ind_tracked1]
            # pts1 = pts1[ind_tracked1]

            # self.draw_matches(img_prev,img,kp0,kp1,matches,'matches.png',PLOT_FIGS=True)   
            R,t = self.visual_odom(pts0,pts1,K)        
            T_c0_c1 = np.eye(4)
            T_c0_c1[:3,:3] = R.T
            T_c0_c1[:3,3] = (-R.T @ t).reshape(-1)
            
            T_m_c0 = np.eye(4)
            T_m_c0[:3,3] = self.poses[i-1,:3]
            T_m_c0[:3,:3] = utils.quat2matrix(self.poses[i-1,3:])

            T_m_c1 = T_m_c0 @ T_c0_c1

            self.poses[i,:3] = T_m_c1[:3,3]
            self.poses[i,3:] = utils.matrix2quat(T_m_c1[:3,:3])

            points_3d = self.triangulate(pts0, pts1, K, T_m_c0, T_m_c1)

            self.points_3d = np.vstack((self.points_3d,points_3d))
            
            self.pts_idx.append(np.ones((pts0.shape[0]),dtype=np.int)*(i-1))
            self.points_2d.append(pts0)            

            self.pts_idx.append(np.ones((pts1.shape[0]),dtype=np.int)*i)
            self.points_2d.append(pts1)

        # self.bundle_adjustment(self.points_2d, self.points_3d, self.pts_idx, self.poses, K)
        self.viz_poses('r')
        self.visualizer.show()




    def bundle_adjustment(self,pts2D_l,pts3D_l,pts_idx,poses,K,plot=True):

        bundle_adjustment = BundleAdjustment2()

        rvecs_l = []
        tvecs_l = []
        for p in poses:
            T_m_c1 = np.eye(4)
            T_m_c1[:3,3] = p[:3]
            T_m_c1[:3,:3] = utils.quat2matrix(p[3:])
            rvec, tvec = utils.T2rt(T_m_c1)
            rvecs_l.append(rvec)
            tvecs_l.append(tvec)

        bundle_adjustment.read_from_data(pts3D_l,pts_idx,pts2D_l,tvecs_l,rvecs_l,K)

        n_cameras = bundle_adjustment.camera_params.shape[0]
        n_points = bundle_adjustment.points_3d.shape[0]

        n = 9 * n_cameras + 3 * n_points
        m = 2 * bundle_adjustment.points_2d.shape[0]

        print("n_cameras: {}".format(n_cameras))
        print("n_points: {}".format(n_points))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))

        _,f0 = bundle_adjustment.fun_init()

        if plot:
            bundle_adjustment.init_viz(0,0,0)
            bundle_adjustment.add_viz('r')
            
        bundle_adjustment.bundle_adjustment_sparsity()
        t0 = time.time()

        res = bundle_adjustment.least_squares()
        t1 = time.time()    

        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        if plot:
            bundle_adjustment.add_viz('c')
            plt.show()

            # plt.plot(f0)
            # plt.plot(res.fun)
            # plt.show()  

    def save_matches(self,pts0_l,pts1_l,matches_l):
        pickle.dump(pts0_l, open(join(self.results_dir,'matches/pts0_l.pkl'), 'wb'))
        pickle.dump(pts1_l, open(join(self.results_dir,'matches/pts1_l.pkl'), 'wb'))
        pickle.dump(pts1_l, open(join(self.results_dir,'matches/pts1_l.pkl'), 'wb'))
        pickle.dump(matches_l, open(join(self.results_dir,'matches/matches_l.pkl'), 'wb'))

    def load_matches(self):
        pts0_l = pickle.load(open(join(self.results_dir,'matches/pts0_l.pkl'), 'rb'))
        pts1_l = pickle.load(open(join(self.results_dir,'matches/pts1_l.pkl'), 'rb'))  
        matches_l = pickle.load(open(join(self.results_dir,'matches/matches_l.pkl'), 'rb'))  
        return pts0_l,pts1_l,matches_l

    def init_viz(self,xp,yp,zp):
        self.visualizer = CameraPoseVisualizer([xp-5, xp+5], [yp-5, yp+5], [zp-2, zp+8])

    def viz_poses(self,color,plot_points=False):

        for i,pose in enumerate(self.poses):

            T = np.eye(4)
            T[:3,3] = pose[:3]            
            T[:3,:3] = utils.quat2matrix(pose[3:])

            self.visualizer.extrinsic2pyramid(T, color, 1)

        if plot_points:
            self.visualizer.plot_points(self.points_3d,color)

    def draw_matches(self,I1,I2,kp1,kp2,matches,fname,PLOT_FIGS=False):
        
        img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
        if PLOT_FIGS:
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()
        
        cv2.imwrite(fname,img)        

    def visual_odom(self,f0,f1,K):
        E, mask = cv2.findEssentialMat(f0, f1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, f0, f1, K)
        return R,t      

    def triangulate(self, f0, f1, K, T_m_c0, T_m_c1):

        T_c0_m = np.eye(4)
        R0 = T_m_c0[:3,:3]
        t0 = T_m_c0[:3,3]
        T_c0_m[:3,:3] = R0.T
        T_c0_m[:3,3] = -R0.T @ t0

        T_c1_m = np.eye(4)
        R1 = T_m_c1[:3,:3]
        t1 = T_m_c1[:3,3]
        T_c1_m[:3,:3] = R1.T
        T_c1_m[:3,3] = -R1.T @ t1

        P0 = K @ T_c0_m[:3,:]
        P1 = K @ T_c1_m[:3,:]
        
        pts_3D = cv2.triangulatePoints(P0, P1, f0.T, f1.T)
        pts_3D = pts_3D[:3,:] / pts_3D[3,:]
        pts_3D = pts_3D.T

        return pts_3D        

    def load_models(self):

        if self.detector == 'ORB' or self.detector == 'SIFT' or self.detector == 'SURF':
            return     
        else:
            self.matcher_model = match_features.load_model(self.detector,self.matcher)
            self.detector_model = detect_features.load_model(self.detector)        

        self.retrieval_model = detect_features.load_model('netvlad')        
        print('Loaded Netvlad model')

    def create_anchor(self, I1):
        
        # detect local features
        fname1 = join(self.results_dir,'local_features','local_%i.h5' % self.counter)
        kp1, des1 = self.feature_detection(I1, self.detector, fname1, model=self.detector_model)

        # detect global features
        fname1_g = join(self.results_dir,'global_features','global_%i.h5' % self.counter)
        self.feature_detection(I1, 'netvlad', fname1_g, model=self.retrieval_model, id='%i.jpg' % self.counter)

        self.counter += 1

        return kp1, des1, fname1
        
    def match_anchors(self,des0,des1,kp0,kp1,fname0,fname1):
        matches = self.feature_matching(des0,des1,self.detector,self.matcher, fname0, fname1, model=self.matcher_model)

        pts0 = np.float32([ kp0[m.queryIdx].pt for m in matches ])
        pts1 = np.float32([ kp1[m.trainIdx].pt for m in matches ])

        matches = np.array([[m.queryIdx,m.trainIdx] for m in matches])

        return matches,pts0,pts1

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

    def feature_matching(self, des1,des2,detector,matcher,fname1=None,fname2=None, model=None):

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

if __name__ == '__main__':
    data_folder = '/home/zaid/datasets/22-08-08-StructuresLab3dSpalling-processed/panorama'
    sfm = SfM(data_folder,create_new_anchors=True)
    sfm.slam()