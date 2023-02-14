import sys
import warnings
from pathlib import Path
import torch
import numpy as np
import os
import subprocess
import logging
logger = logging.getLogger(__name__)

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from third_party.LoFTR.src.loftr import LoFTR, default_cfg
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

    dir_models = {
        'outdoor_ds.ckpt': 'https://raw.githubusercontent.com/MACILLAS/LoFTR/master/weights/outdoor_ds.ckpt',
        'indoor_ds_new.ckpt': 'https://raw.githubusercontent.com/MACILLAS/LoFTR/master/weights/indoor_ds_new.ckpt'
    }

    def _init(self, conf):
        assert conf['weights'] in self.dir_models.keys()

        #cfg = default_cfg
        _default_cfg = deepcopy(default_cfg)
        if str(conf['weights'])[0] == "i":
            _default_cfg['coarse']['temp_bug_fix'] = True

        matcher = LoFTR(config=_default_cfg)

        ckpt_path = os.path.abspath(os.path.join(os.path.realpath(__file__),"../../../third_party/LoFTR", "weights", conf['weights']))

        # Download the checkpoint.
        checkpoint = Path(ckpt_path)
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True)
            link = self.dir_models[conf['weights']]
            cmd = ['wget', link, '-O', str(checkpoint)]
            logger.info(f'Downloading the LoFTR model with `{cmd}`.')
            subprocess.run(cmd, check=True)

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