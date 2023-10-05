import cv2
import requests
import numpy as np
import yaml
import json
import os
import sys
import threading
import io
from PIL import Image


def get_latest(url, project_id=1):
    if url is None:
        return False
    
    endpoint = url + f'/api/v1/project/{project_id}/get_latest'
    response = requests.get(endpoint, stream=True) 
    try:
        img = Image.open(io.BytesIO(response.content))                                 
    except:
        return None
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_args():

    with open('config/params.yaml') as f:
        args = yaml.safe_load(f)
    return args

def main(url):
    
    url = 'http://'+ url+':5000/'

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while True:
        img = get_latest(url)
        if img is not None:
            cv2.imshow('image',img)   
            cv2.waitKey(1)     


if __name__ == '__main__':
    main(sys.argv[1])