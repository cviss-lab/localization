import cv2
import requests
import numpy as np
import yaml
import json
import os
import sys
import threading
import time

def send_query_image(url, I, project_id=1):

    data1 = cv2.imencode('.jpg', I)[1].tobytes()

    files = {'image':data1}
    endpoint = url + f'/api/v1/project/{project_id}/localize'
    try:
        response = requests.post(endpoint, files=files,timeout=0.5) 
        print(response.content)
    except: 
        print('timeout')
        pass



def load_project(url, project_id=1):
    if url is None:
        return False
    
    endpoint = url + f'/api/v1/project/{project_id}/load'
    response = requests.get(endpoint) 
    print(response)
    return True


def main(dataset_dir, frame_rate=1):
    
    url = 'localhost'
    url = 'http://'+ url+':5000/'

    load_project(url)

    img_l = sorted(os.listdir(dataset_dir))
    k = 0

    while True:
        time.sleep(1.0/float(frame_rate))
        if k >= len(img_l):
            k = 0
        f_image = os.path.join(dataset_dir , img_l[k])
        # f_image = os.path.join(dataset_dir , 'rgb', f'{i}.png')
        I = cv2.imread(f_image)

        send_query_image(url, I)

        k += 1


if __name__ == '__main__':
    main(sys.argv[1])