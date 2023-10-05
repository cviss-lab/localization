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
        "--stray_scanner", action="store_true", help="Input is a Stray Scanner dataset"
    )
    parser.add_argument(
        "--mobile_inspector",
        action="store_true",
        help="Input is a Mobile Inspector dataset",
    )
    parser.add_argument(
        "--depth_fill",
        action="store_true",
        help="Use pointcloud from Stray Scanner to fill missing depth values",
    )
    parser.add_argument(
        "--from_lidar",
        action="store_true",
        help="Use pointcloud from Lidar dataset to calculate depth",
    )
    parser.add_argument(
        "--generate_plan",
        type=float,
        default=0,
        help="This flag tells preprocessor to generate floor_plan.png at the designated height (to slice the point "
        "cloud) Assuming the pointcloud ply has already been generated.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the arguments
    input_dir = args.input
    output_dir = args.output
    frame_rate = args.frame_rate
    stray_scanner = args.stray_scanner
    mobile_inspector = args.mobile_inspector
    lidar = args.from_lidar
    depth_fill = args.depth_fill
    depth_max = args.depth_max
    plan_height = args.generate_plan

    # This is so that we can generate plan_view without
    if stray_scanner or mobile_inspector or lidar:
        # remove output directory folders if they exist
        os.makedirs(output_dir, exist_ok=True)
        shutil.rmtree(os.path.join(output_dir, "rgb"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, "depth"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, "report"), ignore_errors=True)

    # create preprocessor object
    preprocessor = PreProcessor(
        input_dir, output_dir, frame_rate=frame_rate, depth_max=depth_max
    )

    t1 = time.time()

    # load stray scanner dataset
    if stray_scanner:
        preprocessor.from_stray_scanner()
    # load mobile inspector dataset
    elif mobile_inspector:
        preprocessor.from_mobile_inspector()
    # load lidar dataset
    elif lidar:
        # preprocessor.from_lidar()
        preprocessor.from_one3d()


    # complete depth maps using pointcloud for missing points
    if depth_fill:
        preprocessor.complete_depth()

    # if cloud.ply exists
    if preprocessor.cloud_exists():
        preprocessor.save_plan_view(plan_height)
        print(preprocessor.loader.load_P_plan(os.path.join(output_dir, "P_plan.txt")))

    t2 = time.time()

    print("Preprocessing data took {} seconds".format(t2 - t1))
