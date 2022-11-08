import sys
import os
from os.path import join, dirname, realpath
import cv2
import numpy as np
import utils
from graphslam.load import load_g2o_se3
from pathlib import Path
import shutil
import h5py
import json
from scipy.spatial.transform import Rotation
import open3d as o3d

# from pose_graph import PoseGraphOptimization
# import g2o

hloc_module=join(dirname(realpath(__file__)),'hloc_toolbox')
sys.path.insert(0,hloc_module)

from hloc_toolbox import match_features, detect_features, search, hloc
from hloc_toolbox.hloc import extract_features, match_features , pairs_from_retrieval
from hloc_toolbox.hloc.utils.parsers import names_to_pair

class GraphSlam():
    def __init__(self,data_folder):
        self.detector = 'superpoint_max'
        self.matcher = 'superglue'
        self.data_folder = join(data_folder, 'images') 
        self.submap_folder = join(data_folder, 'submaps')   

        self.image_dir = join(self.data_folder, 'rgb')
        self.num_images = len([name for name in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, name))]) 

        self.poses = np.loadtxt(join(self.data_folder,'poses.csv'), delimiter=',')    
        self.image_timestamps = np.loadtxt(join(self.data_folder,'timestamps.txt'))    
        self.submap_timestamps = np.loadtxt(join(self.submap_folder,'timestamps.txt'))    
        self.submap_dict = dict()
        self.image_dict = dict()

        self.load_intrinsics()
        # self.build_submaps()
        # self.visual_database()
        # self.find_loops()
        # self.feature_matching()
        # self.construct_graph()
        # self.save_graph()
        self.solve(plot=True)
        # self.save_poses()
        # self.combine_maps()

    def load_intrinsics(self):
        if os.path.exists(join(self.data_folder, 'K1.txt')):
            self.K1 = np.loadtxt(join(self.data_folder, 'K1.txt'))
        elif os.path.exists(join(self.data_folder, 'intrinsics.json')):
            with open(join(self.data_folder, 'intrinsics.json')) as f:
                intrinsics = json.load(f)
            self.K1 = np.array(intrinsics['camera_matrix'])
        else:
            raise Exception('No intrinsics file found')        

    def build_submaps(self):
        current_img = 0
        for i_sm,t_sm in enumerate(self.submap_timestamps):
            img_ind = np.where(self.image_timestamps < t_sm)[0]
            img_ind = img_ind[img_ind >= current_img]
            self.submap_dict[i_sm] = img_ind.tolist()
            if len(img_ind)>0:
                current_img = np.max(img_ind)+1

            for im in img_ind:
                self.image_dict[im] = i_sm
                

    def visual_database(self, num_matched=10, delete_prev=False):

        outputs = Path(self.data_folder)

        images = outputs / 'rgb'
        self.features_dir = outputs / 'features'
        self.f_retrieved_pairs = self.features_dir / 'pairs-netvlad.txt'

        # remove previous results
        if delete_prev:
            shutil.rmtree(str(self.f_retrieved_pairs),ignore_errors=True)
            shutil.rmtree(str(self.features_dir),ignore_errors=True)

        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, self.features_dir)
        pairs_from_retrieval.main(retrieval_path, self.f_retrieved_pairs, num_matched=num_matched)

    def find_loops(self):

        self.f_retrieved_pairs = self.features_dir / 'pairs-netvlad.txt'
        self.f_loop_pairs = self.features_dir / 'pairs-loop.txt'
        self.f_loop_images = self.features_dir / 'images-loop.txt'

        self.loop_pairs = []
        retrieved_pairs = np.genfromtxt(self.f_retrieved_pairs,dtype='str')
        for pair in retrieved_pairs:
            p1 = int(pair[0].split('.')[0])-1
            p2 = int(pair[1].split('.')[0])-1
            if p1 in self.image_dict and p2 in self.image_dict:
                submap1 = self.image_dict[p1]
                submap2 = self.image_dict[p2]
                if submap1 != submap2:
                    if abs(submap1-submap2) > 1:
                        self.loop_pairs.append([pair[0],pair[1]])

        self.loop_images = np.unique(self.loop_pairs)

        np.savetxt(str(self.f_loop_pairs),self.loop_pairs, fmt="%s")
        np.savetxt(str(self.f_loop_images),self.loop_images,  delimiter=" ", fmt="%s")


    def feature_matching(self):

        outputs = Path(self.data_folder)

        images = outputs / 'rgb'
        features_dir = outputs / 'features'

        feature = self.detector
        matcher = self.matcher
        feature_conf = extract_features.confs[feature]
        matcher_conf = match_features.confs[matcher]

        feature_path = extract_features.main(feature_conf, images, features_dir, image_list=self.loop_images)
        match_path = match_features.main(matcher_conf, self.f_loop_pairs, feature_conf['output'], features_dir)
        
        self.features_h5, self.matches_h5 = self.read_matches(feature_path, match_path)
        


    def construct_graph(self):
        self.nodes = self.poses[:,1:].tolist()
        self.edges = []
        
        # build graph
        for i,node in enumerate(self.nodes[:-1]):
            pose1 = self.poses[i  , 1:]
            pose2 = self.poses[i+1, 1:]
            T_m1_c1 = utils.pose2matrix(pose1)
            T_m1_c2 = utils.pose2matrix(pose2)
            T_c1_c2 = utils.T_inv(T_m1_c1) @ T_m1_c2
            rel_pose = utils.matrix2pose(T_c1_c2)
            self.edges.append([i,i+1]+rel_pose.tolist())

        # find loops
        loop_dict = dict()
        for pair in self.loop_pairs:
            p1 = pair[0] # query
            p2 = pair[1] # anchor
            if p1 in loop_dict:
                loop_dict[p1].append(p2)
            else:
                loop_dict[p1] = [p2]

        for p1 in loop_dict.keys():   
            pts2D_all = np.array([]).reshape(0, 2)
            pts3D_all = np.array([]).reshape(0, 3)                     
            for p2 in loop_dict[p1]:
                n1 = int(p1.split('.')[0])-1
                n2 = int(p2.split('.')[0])-1    

                kp1 = self.load_features(p1)
                kp2 = self.load_features(p2)   
                matches = self.load_matches(p1, p2)

                D2 = cv2.imread(join(self.data_folder, 'depth', p2.replace('.jpg','.png')), cv2.IMREAD_UNCHANGED)
                pts1 = np.float32([kp1[m[0]] for m in matches])
                pts2 = np.float32([kp2[m[1]] for m in matches])

                x_c, y_c, z_c = utils.project_2d_to_3d(pts2.T, self.K1, D2, h=0)

                pts3D_c = np.array([x_c, y_c, z_c, np.ones(x_c.shape[0])])

                pose2 = self.poses[n2, 1:]
                T_m1_c2 = utils.pose2matrix(pose2)

                pts3D = T_m1_c2.dot(pts3D_c)
                pts3D = pts3D[:3, :] / pts3D[3, :]

                pts3D = pts3D.T

                idx = np.array([i for i, p in enumerate(pts3D) if not np.any(np.isnan(p))])
                if len(idx) == 0:
                    continue

                pts3D = pts3D[idx]
                pts2D = pts1[idx]

                # pts2D_all = np.vstack([pts2D_all, pts2D])
                # pts3D_all = np.vstack([pts3D_all, pts3D])
                pts2D_all = pts2D
                pts3D_all = pts3D                

                if pts2D_all.shape[0] < 10:
                    print('Loop Rejected..')
                    continue

                retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3D_all, pts2D_all, self.K1, None, flags=cv2.SOLVEPNP_P3P)
                
                if len(inliers)/pts2D_all.shape[0] > 0.8:
                    T_m1_c1 = utils.poses2matrix(tvecs,rvecs)
                    T_c1_c2 = utils.T_inv(T_m1_c1) @ T_m1_c2
                    rel_pose = utils.matrix2pose(T_c1_c2)
                    self.edges.append([n1,n2]+rel_pose.tolist())    
                    print('Found Loop! Adding to Pose graph..')
                else:
                    print('Loop Rejected..')
    
    def save_graph(self, information=np.eye(6)):
        g2o_txt = []
        info = information[np.triu_indices(6)]

        for i,n in enumerate(self.nodes):
            g2o_txt.append('VERTEX_SE3:QUAT {} {} {} {} {} {} {} {}\n'.format(i,*n))

        for e in self.edges:
            g2o_txt.append('EDGE_SE3:QUAT {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n'.format(*e, *info))

        with open(join(self.data_folder,'pose_graph.g2o'),'w') as f:
            for line in g2o_txt:
                f.write(line)

    def save_poses(self):
        self.new_poses = []
        for i,v in enumerate(self.g._vertices):
            self.new_poses.append([i+1]+list(np.array(v.pose)))
        np.savetxt(join(self.data_folder,'new_poses.csv'), self.new_poses, delimiter=',')
        
    def combine_maps(self,voxel_size=0.1):
        pcd_combined = o3d.geometry.PointCloud()
        skip=30
        # self.new_poses = np.loadtxt(join(self.data_folder,'poses.csv'), delimiter=',')
        for p in self.new_poses:
            i = int(p[0])
            if i % skip != 0:
                continue
            pose = p[1:]
            D = cv2.imread(join(self.data_folder, 'depth', str(i)+'.png'), cv2.IMREAD_UNCHANGED)
            X,Y,Z = utils.project_depth_to_cloud(D, self.K1)
            X = X.reshape(-1)
            Y = Y.reshape(-1)
            Z = Z.reshape(-1)
            pts3D_c = np.array([X, Y, Z, np.ones(X.shape[0])])

            T_m1_c2 = utils.pose2matrix(pose)

            pts3D = T_m1_c2.dot(pts3D_c)
            pts3D = (pts3D[:3, :] / pts3D[3, :]).T

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts3D)     
            pcd_combined += pcd.voxel_down_sample(voxel_size)
            print('processed cloud {}'.format(i))
            
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size)
        o3d.visualization.draw_geometries([pcd_combined_down])    
        o3d.io.write_point_cloud(os.path.join(self.data_folder,"output_new.pcd"), pcd_combined_down)        

    def solve(self,plot=False):
        print('Loading Posegraph..')
        g = load_g2o_se3(join(self.data_folder,'pose_graph.g2o'))  
        if plot:
            g.plot(vertex_markersize=1)
        g.calc_chi2()
        g.optimize()
        print('Posegraph Optimized!..')
        if plot:
            g.plot(vertex_markersize=1)
        g.to_g2o(join(self.data_folder,'pose_graph_out.g2o'))
        self.g = g

    def load_features(self, id):
        kp1 = self.features_h5[id]['keypoints'].__array__()
        # kp1 = np.array([cv2.KeyPoint(int(kp1[i,0]),int(kp1[i,1]),3) for i in range(len(kp1))])    
        return kp1

    def load_matches(self, img1, img2):
        pair = names_to_pair(img1, img2)
        matches = self.matches_h5[pair]['matches0'].__array__()
        # matches = [cv2.DMatch(i,m,0) for i,m in enumerate(matches) if m != -1] 
        matches = [[i,m] for i,m in enumerate(matches) if m != -1] 
        return matches       

    def read_matches(self, features_path, matches_path):
        features_h5 = h5py.File(features_path, 'r')
        matches_h5  = h5py.File(matches_path, 'r')
        return features_h5, matches_h5

if __name__ == '__main__':
    data_folder = '/home/zaid/datasets/22-10-11-ParkStBridge-processed'
    sfm = GraphSlam(data_folder)
