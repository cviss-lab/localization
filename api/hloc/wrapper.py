from os.path import dirname, join, realpath
import shutil
import cv2
import numpy as np
from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import h5py
from hloc.utils.parsers import names_to_pair
import shutil

def get_features_path(dataset,detector):
    feature_conf = extract_features.confs[detector]
    fpath = join(dataset, feature_conf['output']+'.h5')
    return fpath

def detect(I1,feature_name,detector,model=None,img_name=None,dataset=None):    

    if img_name is None:
        img_name = 'image.jpg'
        
    cv2.imwrite(join(dataset,img_name),I1)

    dataset = Path(dataset)
    images = dataset 
    output = dataset
    fpath = dataset / feature_name

    feature_conf = extract_features.confs[detector]
    features_file = extract_features.main(feature_conf, images, output, fpath, model=model)   

    if detector != 'netvlad':
        return load_features(features_file, img_name)        
    else:
        return None

def match(f_kp1,f_kp2, img_name1, img_name2, matcher,model=None, dataset=None):  

    dataset = Path(dataset)
    output = dataset
    f_kp1 = dataset / f_kp1
    f_kp2 = dataset / f_kp2

    pairs = dataset / 'pairs.txt'
    np.savetxt(str(pairs),[[img_name1, img_name2]], fmt="%s")

    features_combined = dataset / 'features.h5'
    matcher_conf = match_features.confs[matcher]

    f1 = h5py.File(dataset / f_kp1, 'r')
    f2 = h5py.File(dataset / f_kp2, 'r')

    with h5py.File(features_combined,'w') as f:
        f1.copy(img_name1,f, name=img_name1)
        f2.copy(img_name2,f, name=img_name2)

    matches = match_features.main(
        matcher_conf, pairs, 'features', output, model=model)    

    pair = names_to_pair(img_name1,img_name2)
    match_file = h5py.File(matches, 'r')
    matches1 = match_file[pair]['matches0'].__array__()
    matches = [cv2.DMatch(i,m,0) for i,m in enumerate(matches1) if m != -1] 

    return matches

def search(f_query,f_db,dataset=None,num_matches=5):    
    dataset = Path(dataset)
    f_query = dataset / f_query
    pairs = image_retrieval.main(
                Path(f_query), num_matches,
                db_descriptors=Path(f_db))
    return pairs

def load_features(features_file,img_name):
    feature_h5 = h5py.File(features_file, 'r')
    kp1 = feature_h5[img_name]['keypoints'].__array__()
    kp1 = np.array([cv2.KeyPoint(int(kp1[i,0]),int(kp1[i,1]),3) for i in range(len(kp1))])    
    return kp1

def unwrap_features_db():
    pass
