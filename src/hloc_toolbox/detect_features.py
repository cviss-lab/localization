from os.path import dirname, join, realpath
import shutil
import cv2
import numpy as np
from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import h5py
from hloc.utils.parsers import names_to_pair


def main(I1,fpath,detector,model=None,id=None):
    dataset = join(dirname(realpath(__file__)),'datasets')
    outputs = join(dirname(realpath(__file__)),'outputs')

    shutil.rmtree(dataset, ignore_errors=True)
    shutil.rmtree(outputs, ignore_errors=True)

    try:
        os.mkdir(dataset)
        os.mkdir(outputs)
    except:
        pass

    if id is None:
        id = 'image.jpg'

    cv2.imwrite(join(dataset,id),I1)

    dataset = Path(dataset)
    images = dataset 

    outputs = Path(outputs)
    # pairs = outputs / 'pairs.txt'
    # np.savetxt(str(pairs) ,[['I1.jpg','I2.jpg']], fmt="%s")

    if detector == 'R2D2':
        feature_conf = extract_features.confs['r2d2']
    elif detector == 'SuperPoint':
        feature_conf = extract_features.confs['superpoint_max']
    elif detector == 'D2-Net':
        feature_conf = extract_features.confs['d2net-ss']
    elif detector == 'netvlad':
        feature_conf = extract_features.confs['netvlad']

    features_file = extract_features.main(feature_conf, images, outputs, feature_path=Path(fpath), model=model)   

    # delete temp image
    os.remove(join(dataset,id))

    if detector != 'netvlad':
        return load_features(features_file)        
    else:
        return None

def load_features(features_file,id=None):
    if id is None:
        id = 'image.jpg'
    feature_h5 = h5py.File(features_file, 'r')
    kp1 = feature_h5[id]['keypoints'].__array__()
    kp1 = np.array([cv2.KeyPoint(int(kp1[i,0]),int(kp1[i,1]),3) for i in range(len(kp1))])    
    return kp1

def load_model(detector):

    if detector == 'R2D2':
        feature_conf = extract_features.confs['r2d2']
    elif detector == 'SuperPoint':
        feature_conf = extract_features.confs['superpoint_max']
    elif detector == 'D2-Net':
        feature_conf = extract_features.confs['d2net-ss']
    elif detector == 'netvlad':
        feature_conf = extract_features.confs['netvlad']
        
    model = extract_features.load_model(feature_conf)
    
    return model

if __name__ == "__main__":
    main()