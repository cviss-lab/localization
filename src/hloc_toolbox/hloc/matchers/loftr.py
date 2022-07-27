import sys
import warnings
from pathlib import Path
import torch
import cv2
import numpy as np
import os

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy

class LoFTR(BaseModel):
    default_conf = {
        'weights': 'outdoor_ds.ckpt',
        'max_num_matches': None,
    }

    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        cfg = default_cfg
        matcher = LoFTR(config=default_cfg)

        ckpt_path = os.path.join("weights", conf['weights'])
        matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.net = matcher = matcher.eval().cuda()

    def _forward(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data)

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] = \
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        pred['scores'] = scores
        del pred['confidence']
        return pred