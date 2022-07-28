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

class loftr(BaseModel):
    default_conf = {
        'weights': 'outdoor_ds.ckpt',
        'max_num_matches': 5000,
    }

    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        #cfg = default_cfg
        _default_cfg = deepcopy(default_cfg)
        if str(conf['weights'])[0] == "i":
            _default_cfg['coarse']['temp_bug_fix'] = True

        matcher = LoFTR(config=_default_cfg)

        ckpt_path = os.path.join("./hloc_toolbox/third_party/LoFTR", "weights", conf['weights'])

        matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
        matcher = matcher.eval().cuda()
        self.net = matcher

    def _forward(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                self.net(data)
                pred = data
        scores = pred['mconf']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['mkpts0_f'], pred['mkpts1_f'] = \
                pred['mkpts0_f'][keep], pred['mkpts1_f'][keep]
            scores = scores[keep]

        pred['mconf'] = scores
        #del pred['confidence']
        return pred