import argparse
import logging
from os import mkdir, path
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import shutil
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary
from .utils.io import list_h5_names


def main(descriptors, num_matched,output=None,
        db_descriptors=None):
    logging.info('Extracting image pairs from a retrieval database.')

    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_names = list(name2db.keys())
    query_names = list_h5_names(descriptors)

    if num_matched > len(db_names):
        num_matched = len(db_names)

    def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
        if name2idx is None:
            with h5py.File(str(path), 'r') as fd:
                desc = [fd[n][key].__array__() for n in names]
        else:
            desc = []
            for n in names:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())
        return torch.from_numpy(np.stack(desc, 0)).to(device).float()

    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()
    scores = sim[torch.arange(len(query_names)), topk].cpu().numpy().reshape(-1)
    
    pairs = []
    for query, indices in zip(query_names, topk):
        for i in indices:
            pair = (query, db_names[i])
            pairs.append(pair)

    if output is not None:
        with open(output, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    return pairs, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
