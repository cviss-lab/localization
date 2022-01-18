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

from src.hloc_toolbox import match_features, detect_features

PLOT_FIGS = False

def main():
    
    dataset_dir = join(dirname(realpath(__file__)),'datasets/markers')
    anchors_dir = join(dirname(realpath(__file__)),'datasets/anchors')

    results_dir = join(dirname(realpath(__file__)),'results')

    shutil.rmtree(results_dir,ignore_errors=True)
    os.mkdir(results_dir)
    os.mkdir(join(results_dir,'matches'))
    os.mkdir(join(results_dir,'reproj'))

    marker_list = sorted(os.listdir(dataset_dir))
    anchor_list = sorted(os.listdir(join(anchors_dir,'color')))

    # marker_list = ['c']

    features_types = ['ORB','SIFT','R2D2','SuperPoint+NN','SuperPoint+superglue']

    error_features = pd.DataFrame(index=features_types,columns=['Positional Error','Outliers','Time','Matches'])

    for features_type in features_types:

        error_markers = []
        eval_time = []
        num_matches = []
        anchors_data = []

        for anchor in range(len(anchor_list)):
            I1 = cv2.imread(join(anchors_dir,'color',str(anchor)+'.jpg'))
            if features_type == 'ORB' or features_type == 'SIFT' or features_type == 'SURF':
                kp1, des1 = feature_detection(I1,detector=features_type)
            else:
                f_kp1 = join(results_dir,'kp'+str(anchor)+'.h5')
                if os.path.exists(f_kp1):
                    os.remove(f_kp1)                
                kp1 = detect_features.main(I1,f_kp1,features_type)    
                des1 = None
            anchors_data.append([kp1,des1])

        for marker_name in marker_list:
            input_dir = join(dataset_dir,marker_name) 
            query_files = os.listdir(join(input_dir,'query'))

            for query_id,query_file in enumerate(query_files):
                I2 = cv2.imread(join(input_dir,'query',query_file))
                K2 = np.loadtxt(join(input_dir,'intrinsics.txt'))

                t1 = time.time()
                found_anchor = False

                if features_type == 'ORB' or features_type == 'SIFT' or features_type == 'SURF':
                    kp2, des2 = feature_detection(I2,detector=features_type)
                else:
                    f_kp2 = join(results_dir,'kp_q.h5')
                    if os.path.exists(f_kp2):
                        os.remove(f_kp2)
                    kp2 = detect_features.main(I2,f_kp2,features_type)  
                    des2 = None

                n_inliers = 0
                for anchor in range(len(anchor_list)):

                    kp1, des1 = anchors_data[anchor]
                    if features_type == 'ORB' or features_type == 'SIFT' or features_type == 'SURF':
                        matches = feature_matching(des1,des2)
                    else:
                        f_kp1 = join(results_dir,'kp'+str(anchor)+'.h5')
                        f_kp2 = join(results_dir,'kp_q.h5')
                        matches = match_features.main(f_kp1,f_kp2,features_type)


                    if len(matches) < 20:
                        continue

                    D1 = cv2.imread(join(anchors_dir,'depth',str(anchor)+'.png'),cv2.IMREAD_UNCHANGED)
                    K1 = np.loadtxt(join(anchors_dir,'intrinsics.txt'))

                    pts1 = np.float32([ kp1[m.queryIdx].pt for m in matches ])
                    pts2 = np.float32([ kp2[m.trainIdx].pt for m in matches ])

                    x,y,z = project_2d_to_3d(pts1.T,K1,D1)
                    pts3D = np.array([x,y,z]).T            
                    
                    idx = np.array([i for i,p in enumerate(pts3D) if not np.any(np.isnan(p))])
                    if len(idx) < 10:
                        continue

                    pts3D = pts3D[idx]
                    pts2D = pts2[idx]

                    retval,rvecs,tvecs,inliers=cv2.solvePnPRansac(pts3D, pts2D, K2, None,flags=cv2.SOLVEPNP_P3P)
                    if not retval or len(inliers)/len(pts2D) < 0.5:
                        continue

                    found_anchor = True
                    if n_inliers < len(inliers):

                        t2 = time.time()

                        # find relocalized pose of query image relative to robot camera

                        R = cv2.Rodrigues(rvecs)[0]
                        t = tvecs
                        P = np.dot(K2, np.hstack([R, t]))      

                        I1 = cv2.imread(join(anchors_dir,'color',str(anchor)+'.jpg'))

                        marker_length = 0.12

                        fname1 = '_'.join([features_type,marker_name,str(query_id)])+'_reproj.png'
                        fname1 = join(results_dir,'reproj',fname1)

                        fname2 = '_'.join([features_type,marker_name,str(query_id)])+'_matches.png'
                        fname2 = join(results_dir,'matches',fname2)
                        
                        error = reproj_error(P,I1,D1,K1,I2,K2,marker_length,features_type,query_id,marker_name,fname1)   

                        if n_inliers == 0:
                            error_markers.append(error)                    
                            eval_time.append(t2-t1)
                            num_matches.append(len(inliers))
                        else:
                            error_markers[-1] = error                  
                            eval_time[-1] = t2-t1
                            num_matches[-1] = len(inliers)

                        n_inliers = len(inliers)
                        
                        draw_matches(I1,I2,kp1,kp2,matches,fname2)

                if not found_anchor:
                    error_markers.append(-1)

        error_markers = np.array(error_markers)
        error_features.loc[features_type]['Positional Error'] = 100*np.mean(error_markers[error_markers!=-1])
        error_features.loc[features_type]['Outliers'] = np.sum(error_markers==-1)
        error_features.loc[features_type]['Time'] = np.mean(eval_time)
        error_features.loc[features_type]['Matches'] = np.mean(num_matches)

    error_features.to_csv(join(results_dir,'results.csv'))

def reproj_error(P2,I1,D1,K1,I2,K2,marker_length,features_type,query_id,marker_name, fname):

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

    x1,y1 = np.int32(m2_proj[:,0])
    x2,y2 = np.int32(m2_proj[:,1])
    x3,y3 = np.int32(m2_proj[:,2])
    x4,y4 = np.int32(m2_proj[:,3])

    import copy
    img = copy.copy(I2)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=3)
    cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), thickness=3)
    cv2.line(img, (x4, y4), (x1, y1), (0, 255, 0), thickness=3)
    
    cv2.imwrite(fname,img)

    if PLOT_FIGS:
        plt.imshow(cv2.cvtColor(I2,cv2.COLOR_RGB2BGR)),plt.show()        
    
    return reproj_error_scaled

def draw_matches(I1,I2,kp1,kp2,matches,fname):
    
    img = cv2.drawMatches(I1,kp1,I2,kp2,matches,None,flags=2)
    if PLOT_FIGS:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_RGB2BGR)),plt.show()
    
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

    return matches     


if __name__ == '__main__':
    main()    
    
