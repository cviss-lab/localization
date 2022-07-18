import utils
import json
import numpy as np
# sfm_file = '/home/jp/Desktop/Rishabh/Handheld/1m_debug6/0_6_less_img_ultra/reconstruction_global/sfm_data.json'

def return_T_M2_C2(sfm_file):
    # sfm_file = '/home/jp/Desktop/Rishabh/Handheld/1m_debug6/0_6_less_img_ultra/reconstruction_global/sfm_data.json'
    with open(sfm_file, 'r') as f:
        sfm = json.load(f)

    T_m2_c2_list = []
    for i in range(len(sfm['views'])):
        # print(sfm['views'])
        rotation = np.array(sfm['extrinsics'][i]['value']['rotation'])
        centers = np.array(sfm['extrinsics'][i]['value']['center'])
        T_m2_c2 = np.eye(4)
        R = rotation.T
        # t = R.dot(-centers).reshape(-1, 1)
        C = centers.reshape((-1, 1))
        # Tm2_c2 = np.hstack([R, t])
        T_m2_c2[:3, :3] = R
        T_m2_c2[:3, 3] = C.reshape(-1)
        T_m2_c2_list.append(T_m2_c2)
    return T_m2_c2_list
print("TEST")
