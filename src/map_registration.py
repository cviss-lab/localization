import open3d as o3d
import numpy as np
import copy


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


print("1. Load two point clouds and show initial pose")

# Load the source map (smaller map)
source = o3d.io.read_point_cloud(
    "/home/jp/Desktop/Rishabh/Handheld/localisation_structures_hl2/0_6_55_img_ultra/reconstruction_global/Thresh_colorized.ply")
# Load the target map (larger map)
target = o3d.io.read_point_cloud("/home/jp/Desktop/Rishabh/Handheld/localisation_test_data/r3live_output/rgb_pt.pcd")

# Add the initial transformation (if available, otherwise Identity matrix
trans_init = np.asarray([[0.54186, -0.83509, -0.09491, 3.48543],
                         [0.83752, 0.54595, -0.02213, 1.00575],
                         [0.07030, -0.06750, 0.99524, 0.32907],
                         [0.00000, 0.00000, 0.00000, 1.00000]])

source.transform(trans_init)
source.estimate_normals()
target.estimate_normals()

# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017
voxel_radius = [0.04, 0.02, 0.01]
max_iter = [50, 30, 14]
current_transformation = np.identity(4)
print("3. Colored point cloud registration")
for scale in range(3):
    iter = max_iter[scale]
    radius = voxel_radius[scale]
    print([iter, radius, scale])

    print("3-1. Downsample with a voxel size %.2f" % radius)
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    print("3-2. Estimate normal.")
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    print("3-3. Applying colored point cloud registration")
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=iter))
    current_transformation = result_icp.transformation
    print(result_icp)

# Saves the 4x4 transform as a txt file
np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/T_colored_icp.txt", result_icp.transformation)
np.savetxt("/home/jp/Desktop/Rishabh/Handheld/localisation_structures_ig4/T_colored_icp_total.txt", np.dot(result_icp.transformation,trans_init))
draw_registration_result_original_color(source, target,
                                        result_icp.transformation)

print(result_icp.transformation)
