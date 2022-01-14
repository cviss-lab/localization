#!/usr/bin/env python3  
import os
from os.path import join, dirname, realpath
import shutil
import sys
import numpy as np
import cv2
import pandas as pd
from src.utils import *
from matplotlib import pyplot as plt
import time
hloc_module=join(dirname(realpath(__file__)),'src/hloc_toolbox')
sys.path.insert(0,hloc_module)

from src.hloc_toolbox import match_features

PLOT_FIGS = False

def main():
    
    dataset_dir = join(dirname(realpath(__file__)),'datasets/markers')
    
    results_dir = join(dirname(realpath(__file__)),'results')

    shutil.rmtree(results_dir,ignore_errors=True)
    os.mkdir(results_dir)
    os.mkdir(join(results_dir,'matches'))

    marker_list = sorted(os.listdir(dataset_dir))
    # marker_list = ['c']

    features_types = ['ORB','SIFT','R2D2','SuperPoint+NN','SuperPoint+superglue']
    error_features = pd.DataFrame(index=features_types,columns=['Positional Error','Outliers','Time','Matches'])

    for features_type in features_types:

        error_markers = []
        eval_time = []
        num_matches = []

        for marker_name in marker_list:
            input_dir = join(dataset_dir,marker_name)

            I1 = cv2.imread(join(input_dir,'image.jpg'))
            D1 = cv2.imread(join(input_dir,'depth.png'),cv2.IMREAD_UNCHANGED)
            K1 = np.loadtxt(join(input_dir,'K1.txt'))
            K2 = np.loadtxt(join(input_dir,'K2.txt'))
            
            query_files = os.listdir(join(input_dir,'query'))

            for query_id,query_file in enumerate(query_files):
                I2 = cv2.imread(join(input_dir,'query',query_file))

                t1 = time.time()
                if features_type == 'ORB' or features_type == 'SIFT' or features_type == 'SURF':
                    kp1, des1 = feature_detection(I1,detector=features_type)
                    kp2, des2 = feature_detection(I2,detector=features_type)
                    matches = feature_matching(des1,des2)
                else:
                    matches, kp1, kp2 = match_features.main(I1,I2,features_type)

                num_matches.append(len(matches))
                if len(matches) > 20:
                    pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
                    pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

                    x,y,z = project_2d_to_3d(pts1.T,K1,D1)
                    pts3D = np.array([x,y,z]).T            
                    
                    idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
                    pts3D = pts3D[idx]
                    pts2D = pts2[idx]

                    if len(idx) < 10:
                        error_markers.append(-1)
                            
                    retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D, pts2D, K2, None,flags=cv2.SOLVEPNP_P3P)
                    
                    # find relocalized pose of query image relative to robot camera

                    R = cv2.Rodrigues(rvecs)[0]
                    t = tvecs
                    P = np.dot(K2, np.hstack([R, t]))      

                    t2 = time.time()
                    eval_time.append(t2-t1)

                    if PLOT_FIGS:
                        draw_matches(I1,I2,kp1,kp2,matches)    
                    
                    marker_length = 0.12
                    error = reproj_error(P,I1,D1,K1,I2,K2,marker_length,features_type,query_id,marker_name)   
                    error_markers.append(error)
                else:
                    error_markers.append(-1)

                fname = '_'.join([features_type,marker_name,str(query_id)])+'_matches.png'
                fname = join(results_dir,'matches',fname)
                draw_matches(I1,I2,kp1,kp2,matches,plot=False,fname=fname)

        error_markers = np.array(error_markers)
        error_features.loc[features_type]['Positional Error'] = 100*np.mean(error_markers[error_markers!=-1])
        error_features.loc[features_type]['Outliers'] = np.sum(error_markers==-1)
        error_features.loc[features_type]['Time'] = np.mean(eval_time)
        error_features.loc[features_type]['Matches'] = np.mean(num_matches)

    f_results = open(join(results_dir,'results.txt'), 'w')
    f_results.write(error_features.to_string())
    f_results.close()

def reproj_error(P2,I1,D1,K1,I2,K2,marker_length,features_type,query_id,marker_name):

    m1 = detect_markers(I1,xy_array=True)
    m2 = detect_markers(I2,xy_array=True)
    id1 = m1[0]['id']
    id2 = m2[0]['id']

    if m1 is None or m2 is None:
        return
    else:
        m1 = m1[0]['coords']
        m2 = m2[0]['coords']            

    x,y,z = project_2d_to_3d(m1,K1,D1)
    m_3D = np.array([x,y,z]).T

    m2_proj = ProjectToImage(P2,m_3D.T)
    reproj_error = np.sqrt(np.sum((m2-m2_proj)[0,:]**2 + (m2-m2_proj)[1,:]**2))
    # print("Reprojection Error: %s" % reproj_error)

    m2_scaled = np.array([[0,0],[marker_length,0],[marker_length,marker_length],[0,marker_length]])
    H = cv2.findHomography(m2.T,m2_scaled)[0] 

    m2_proj_scaled = cv2.perspectiveTransform(m2_proj.T.reshape(-1,1,2),H).reshape(-1,2)

    diff = (m2_proj_scaled - m2_scaled).T
    reproj_error_scaled = np.mean(np.sqrt(diff[0,:]**2 + diff[1,:]**2))

    if id1 != id2:
        for _ in range(4):
            m2_proj_scaled = np.array([m2_proj_scaled[i,:] for i in [1,2,3,0]])
            diff = (m2_proj_scaled - m2_scaled).T
            reproj_error_scaled2 = np.mean(np.sqrt(diff[0,:]**2 + diff[1,:]**2))
            if reproj_error_scaled2 < reproj_error_scaled:
                reproj_error_scaled = reproj_error_scaled2

    print("%s, %s, %i Positional Error: %s cm" % (features_type,marker_name, query_id, reproj_error_scaled*100))

    if PLOT_FIGS:
        x1,y1 = np.int32(m2_proj[:,0])
        x2,y2 = np.int32(m2_proj[:,1])
        x3,y3 = np.int32(m2_proj[:,2])
        x4,y4 = np.int32(m2_proj[:,3])

        cv2.line(I2, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv2.line(I2, (x2, y2), (x3, y3), (0, 255, 0), thickness=3)
        cv2.line(I2, (x3, y3), (x4, y4), (0, 255, 0), thickness=3)
        cv2.line(I2, (x4, y4), (x1, y1), (0, 255, 0), thickness=3)
        
        plt.imshow(cv2.cvtColor(I2,cv2.COLOR_RGB2BGR)),plt.show()        
    
    return reproj_error_scaled

def draw_matches(I1,I2,kp1,kp2,matches,plot=True,fname='matches.png'):
    
    img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
    if plot:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()
    else:
        cv2.imwrite(fname,img)

def feature_detection(I,detector='SIFT'):
    
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

    return kp, des

def feature_matching(des1,des2):
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
    main()    
    
