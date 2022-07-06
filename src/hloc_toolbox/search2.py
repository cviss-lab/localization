from pathlib import Path
from hloc import extract_features, image_retrieval, match_features
import os
import argparse

def main(dataset,outputs, num_matches, build_database):
    dataset = Path(dataset)
    images = dataset / 'images/'
    query = dataset / 'query/'

    outputs = Path(outputs)
    loc_pairs = outputs / f'pairs-query-netvlad{num_matches}.txt'  # top-k retrieved by NetVLAD
    features_path_query = outputs / 'global-feats-query.h5'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['netvlad']

    features_path = Path.with_suffix(outputs / retrieval_conf['output'],'.h5')
    ## If previous descriptors file exists, delete it ##
    if build_database:
        if os.path.isfile(features_path):
            os.remove(features_path)
        global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    else:
        global_descriptors = Path(features_path)

    if num_matches == 0:
        return

    ## If previous query descriptors file exists, delete it ##
    if os.path.isfile(features_path_query):
        os.remove(features_path_query)

    global_descriptors_q = extract_features.main(retrieval_conf, query, outputs,
                                                feature_path=features_path_query)

    image_retrieval.main(
        global_descriptors_q, loc_pairs, num_matches,
        images, query, db_descriptors=global_descriptors)

    # feature_conf = extract_features.confs['superpoint_max']
    # matcher_conf = match_features.confs['superglue']
    # features = extract_features.main(feature_conf, images, outputs)
    # loc_matches = match_features.main(
    #     matcher_conf, loc_pairs, feature_conf['output'], outputs)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument('--num_matches', type=int, required=True)
    parser.add_argument('--build_database', default=False, action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)