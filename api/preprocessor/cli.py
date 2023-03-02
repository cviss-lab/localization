#!/usr/bin/python3
import shutil
import os
import argparse
import time

from preprocessor import PreProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to input directory"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--frame_rate",
        "-f",
        type=int,
        default=2,
        help="Frame rate to use from Stray Scanner dataset",
    )
    parser.add_argument(
        "--depth_max",
        "-d",
        type=float,
        default=5,
        help="Maximum depth from Stray Scanner dataset",
    )

    parser.add_argument(
        "--voxel",
        "-v",
        type=float,
        default=0.015,
        help="Voxel size",
    )    

    parser.add_argument(
        "--mobile_inspector",
        action="store_true",
        help="Input is a Mobile Inspector dataset",
    )


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the arguments
    input_dir = args.input
    output_dir = args.output
    frame_rate = args.frame_rate
    mobile_inspector = args.mobile_inspector
    depth_max = args.depth_max
    voxel = args.voxel


    # This is so that we can generate plan_view without
    if mobile_inspector:
        # remove output directory folders if they exist
        os.makedirs(output_dir, exist_ok=True)
        shutil.rmtree(os.path.join(output_dir, "rgb"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, "depth"), ignore_errors=True)

    # create preprocessor object
    preprocessor = PreProcessor(
        input_dir, output_dir, frame_rate=frame_rate, depth_max=depth_max
    )

    t1 = time.time()

    if mobile_inspector:
        preprocessor.from_mobile_inspector(voxel=voxel)

    t2 = time.time()

    print("Preprocessing data took {} seconds".format(t2 - t1))
