from os.path import dirname, join, realpath
import shutil
import cv2
import numpy as np
from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import h5py
from hloc.utils.parsers import names_to_pair


def main(f_query,fdir_db,num_matches=5):
    db_list = [Path(os.path.join(fdir_db,f)) for f in os.listdir(fdir_db)]
    pairs = image_retrieval.main(
                Path(f_query), num_matches,
                db_descriptors=db_list)
    return pairs

if __name__ == "__main__":
    main()