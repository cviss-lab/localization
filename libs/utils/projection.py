import cv2
from PIL import Image
import numpy as np
# from shapely import Polygon

# from loader import *
import copy
from scipy.spatial.transform import Rotation
from .render_depthmap import VisOpen3D
import sys
try:
    import open3d as o3d
except:
    pass

def calc_centroid(points):
    """
    Takes a list of tuples (points) returns the average (X, Y, Z)
    :param points: [(X, Y, Z), (X, Y, Z), ...]
    :return: tuple (X, Y, Z)
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    return tuple(centroid)


def calc_gnd_plane(gnd_points):
    """
    Takes user picked points converts to plane equation
    :param gnd_points: list of tuples [(X, Y, Z), (X, Y, Z), (X, Y, Z)]
    :return: coefficients of the plane tuples (a, b, c, d)
    """
    # a1 = x2 - x1
    a1 = gnd_points[1][0] - gnd_points[0][0]
    # b1 = y2 - y1
    b1 = gnd_points[1][1] - gnd_points[0][1]
    # c1 = z2 - z1
    c1 = gnd_points[1][2] - gnd_points[0][2]
    # a2 = x3 - x1
    a2 = gnd_points[2][0] - gnd_points[0][0]
    # b2 = y3 - y1
    b2 = gnd_points[2][1] - gnd_points[0][1]
    # c2 = z3 - z1
    c2 = gnd_points[2][2] - gnd_points[0][2]

    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    # d = (- a * x1 - b * y1 - c * z1)
    d = -a * gnd_points[0][0] - b * gnd_points[0][1] - c * gnd_points[0][2]

    return (a, b, c, d)


def slice_pc(pointcloud, gnd_plane=None, height=0.5, thickness=0.7, keep_bottom=False):
    """

    :param pointcloud: o3d pointcloud object
    :param gnd_plane: tuple of scalar plane coefficients
    :param height: height of slice
    :param thickness: slice thickness
    :return: o3d pointcloud object, transformation
    """

    if 'open3d' not in sys.modules:
        raise ImportError('Open3D not installed')

    if gnd_plane is not None:
        T_p_m = find_plane_transformation(gnd_plane, h=height)
    else:
        T_p_m = np.eye(4)

    pointcloud2 = copy.copy(pointcloud)
    pointcloud2.transform(T_p_m)

    # height = np.array(pointcloud2.points)[:,1].mean()

    # slice pointcloud
    if keep_bottom:
        axis = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-np.inf, -np.inf, -np.inf),
            max_bound=(np.inf, np.inf, thickness / 2),
        )
    else:
        axis = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-np.inf, -np.inf, -thickness / 2),
            max_bound=(np.inf, np.inf, thickness / 2),
        )

    pointcloud_cropped = pointcloud2.crop(axis)

    return pointcloud_cropped, T_p_m


def create_plan_view(
    pointcloud, gnd_plane, height=2.5, thickness=0.7, colored=True, w=1920, h=1080
):
    """
    Creates a plan view image from a pointcloud
    :param pointcloud: o3d pointcloud object
    :param gnd_plane: tuple of scalar plane coefficients
    :param height: height of slice
    :param thickness: slice thickness
    :param colored: boolean
    :param w: int image width
    :param h: int image height
    :return: PIL Image, projection matrix (3x4) from map to image
    """

    pc, T_p_m = slice_pc(
        pointcloud, gnd_plane, thickness=thickness, height=height, keep_bottom=False
    )

    xyz = np.asarray(pc.points)
    x = xyz[:, 0]
    y = xyz[:, 1]

    xc = np.mean(x)
    yc = np.mean(y)

    # calculate scale factor
    sx = w / (x.max() - x.min()) / 2
    sy = h / (y.max() - y.min()) / 2
    s = np.min([sx, sy])

    # orthographic projection matrix
    # fmt: off
    K = np.array(
        [
            [s, 0, 0, w / 2 - s * xc],
            [0, s, 0, h / 2 - s * yc],
            [0, 0, 0, 1]
        ]
    )
    # fmt: on

    # project 3d points to 2d image plane
    I = binary_image(xyz.T, K, w, h)

    H, ccw_angle = find_homography(I)

    # # project image again after calculating new K
    if colored:
        pc, _ = slice_pc(pointcloud, gnd_plane, height=height, keep_bottom=True)
        I, P = colored_image(pc, s, T_p_m, xc, yc, w, h, upscale=2, H=H)
    else:
        K = H.dot(K)
        P = K.dot(T_p_m)
        I = binary_image(xyz.T, K, w, h)

    img = Image.fromarray(I)

    return img, P


def binary_image(pts_3d, P, w, h):
    """
    creates a binary mask image by projecting 3d points onto image plane

    :param pts_3d: numpy array (3, n) where n is the number of points
    :param P: 3x4 from map frame to plane frame
    :param w: int image width
    :param h: int image height
    :return: image
    """
    uv = world_pt_to_plan_img(pts_3d, P)
    u = uv[0, :]
    v = uv[1, :]

    ind = np.logical_and(np.logical_and(u < w, v < h), np.logical_and(u >= 0, v >= 0))
    u = u[ind]
    v = v[ind]

    I = np.zeros((h, w), dtype=np.uint8)
    I[v, u] = 255

    return I


def colored_image(pc, s, T_p_m, xc, yc, w, h, upscale=1, fov=1, H=None):
    """
    creates a color image by projecting 3d points onto image plane using open3d
    :param pc: o3d pointcloud object
    :param s: scale factor
    :param T_p_m: transformation from map to plane
    :param xc: x center
    :param yc: y center
    :param w: int image width
    :param h: int image height
    :param upscale: int upsampling factor
    :param fov: float field of view
    :return: image, projection matrix (3x4) from map to image
    """

    w2 = w * upscale
    h2 = h * upscale
    s2 = s * upscale

    vis = VisOpen3D(width=w2, height=h2, visible=False)
    vis.add_geometry(pc)

    ratio = np.tan(fov * np.pi / 180 / 2)
    fx = w2 / (2 * ratio)
    fy = fx
    cx = w2 / 2
    cy = h2 / 2
    # fmt: off
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    )
    # fmt: on
    # K2 = copy.copy(K)
    # K2[:2,:2] = K2[:2,:2]*upscale

    l = w2 / 2 / s2
    z = l / ratio

    p = [xc, yc, -z]
    q = euler2quat([0, 0, 0])
    pose_view = list(p) + list(q)
    T_p_view = pose2matrix(pose_view)
    T_view_p = T_inv(T_p_view)
    T_view_m = T_view_p.dot(T_p_m)

    vis.update_view_point(K, T_view_p, point_size=upscale, znear=0.01, zfar=1e9)
    img = vis.capture_screen_float_buffer(show=False)
    img = np.array(np.asarray(img) * 255, dtype=np.uint8)
    P = K.dot(T_view_m[:3, :])

    if H is not None:
        M = np.eye(3)
        M[:2, :2] = M[:2, :2] / upscale
        H = H.dot(M)
        img = cv2.warpPerspective(img, H, (w, h))
        P = H.dot(P)

    return img, P


def find_plane_transformation(plane, h=0):
    """
    calculate transformation from map frame to plane frame (T^p_m)

    :param pointcloud: plane equation (a,b,c,d) and height to cut plane (h)
    :return: transformation (4x4 matrix) from map frame to plane frame (T^p_m)
    """

    a, b, c, d = plane
    nz = np.array([a, b, c]) / np.linalg.norm([a, b, c])

    # make sure nz is facing upwards
    if np.dot([0, 1, 0], nz) < 0:
        nz = -1 * nz

    # find p0 at height (h) above plane in nz direction at xy=(0,0)
    if c == 0:
        p0 = h * nz
    else:
        p0 = [0, 0, -d / c] + h * nz

    # calculate nx from cross product of nz and one of the axis
    if abs(nz.dot([1, 0, 0])) < 1:
        nx = np.cross(nz, [1, 0, 0])
    elif abs(nz.dot([0, 1, 0])) < 1:
        nx = np.cross(nz, [0, 1, 0])
    elif abs(nz.dot([0, 0, 1])) < 1:
        nx = np.cross(nz, [0, 0, 1])

    nx = nx / np.linalg.norm(nx)
    ny = np.cross(nx, nz)

    R = np.array([nx, ny, nz])

    T_p_m = np.eye(4)
    T_p_m[:3, :3] = R
    T_p_m[:3, 3] = R.dot(-p0)

    return T_p_m


def world_pt_to_plan_img(
    pts_3d, P, return_tuples=False, return_ints=True, normalize_w=True
):
    """
    Takes 3d points (pts_3d) in world coordinates and projects them to image coordinates
    using P transformation (3x4 matrix) from map frame to plane frame
    :param pts_3d: numpy array (3, n) where n is the number of points
    :param P: 3x4 from map frame to plane frame
    :param return_tuples: bool return output as list of tuples
    :param return_ints: bool convert output coordinates to integers
    :param normalize_w: bool normalize img_coords by w coordinate
    :return: list of tuples corresponding to the (x, y) coordinates of each 3d point
    """
    # Get number of columns (number of vectors)
    cols = pts_3d.shape[1]

    # Array to be added as row
    row_to_be_added = np.ones(cols)

    # Adding row to numpy array
    pts_3d_homo = np.vstack((pts_3d, row_to_be_added))

    img_coords = np.matmul(P, pts_3d_homo)

    if normalize_w:
        img_coords = img_coords[:2, :] / img_coords[2, :]
    else:
        img_coords = img_coords[:2, :]

    if return_ints:
        img_coords = np.array(img_coords, dtype=int)

    # Convert and return as list of tuples
    if return_tuples:
        img_coords = arr_to_list_of_tuples(img_coords)

    return img_coords


def cloud_to_depth(pcd, K, pose, w, h, point_size=2, s=1):
    """
    uses o3d pointcloud, camera matrix, pose, and image size to calculate depth image
    :param pointcloud: o3d pointcloud object
    :param K: camera matrix (3x3)
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :param w: image width
    :param h: image height
    :return: np array depth image (hxw)
    """

    vis = VisOpen3D(width=int(s * w), height=int(s * h), visible=False)
    vis.add_geometry(pcd)
    T_m_c = pose2matrix(pose)
    T_c_m = T_inv(T_m_c)
    Ks = copy.copy(K)
    Ks = s * K
    Ks[2, 2] = 1
    vis.update_view_point(Ks, T_c_m, znear=1e-4, zfar=65.536, point_size=point_size)
    depth = vis.capture_depth_float_buffer(show=False)    
    depth = np.array(np.asarray(depth) * 1000, dtype=np.uint16)

    # depth = vis.capture_screen_float_buffer(show=True)
    # depth = np.asarray(depth)*255

    return depth


def depth_to_cloud(color, depth, K, pose, save_path=None):
    """
    uses color image, depth image, camera matrix, pose to calculate o3d pointcloud
    :param color: np array color image (hxwx3)
    :param depth: np array depth image (hxw)
    :param K: camera matrix (3x3)
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :param save_path: path to save pointcloud as .ply file
    :return: o3d pointcloud object
    """

    if 'open3d' not in sys.modules:
        raise ImportError('Open3D not installed')    

    height, width, _ = color.shape
    depth_height, depth_width = depth.shape
    s = depth_width / width

    fx = K[0, 0] * s
    fy = K[1, 1] * s
    cx = K[0, 2] * s
    cy = K[1, 2] * s

    # depth = cv2.resize(depth,(width,height), cv2.INTER_NEAREST)
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = cv2.resize(color, (depth_width, depth_height), cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    depth = o3d.geometry.Image(np.float32(depth))
    color = o3d.geometry.Image(color)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=1e9, convert_rgb_to_intensity=False
    )

    intr = o3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, cx, cy)

    T_m_c = pose2matrix(pose)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr, T_m_c)

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pcd)

    return pcd

def check_if_ccw(pts_2d):
    """
    checks if points are counter clockwise
    :param: numpy array (2, n) where n is the number of points
    :return bool:
    """
    y = pts_2d[0, :]
    x = pts_2d[1, :]
    vec1 = np.array([x[1] - x[0], y[1] - y[0]])
    vec2 = np.array([x[2] - x[1], y[2] - y[1]])
    cross = np.cross(vec1, vec2)
    return cross > 0
    
    

def project_3d_to_2d(pts_3d, K, pose):
    """
    Takes 3d points (pts_3d) in world coordinates and projects them to image coordinates.
    :param pts_3d: numpy array (3, n) where n is the number of points
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :param K: array (3x3) camera matrix
    :return: numpy array (2, n) where n is the number of points
    """

    if pts_3d.shape[1] == 0:
        return np.empty((2, 0))

    tvec, rvec = pose2vecs(pose)
    pts_2d = (
        cv2.projectPoints(np.float32(pts_3d), rvec, tvec, K, None)[0].reshape(-1, 2).T
    )

    return pts_2d


def filter_occlusion(pts_3d, pts_2d, depth, K, pose, w, h, occlusion_tol=1.0):
    """
    Checks 2d-3d points pairs for occlusion
    :param pts_3d: numpy array (3, n) where n is the number of points
    :param pts_2d: numpy array (2, n) where n is the number of points
    :param depth: array depth image used to check for occlusion using depth buffer method
    :param K: array (3x3) camera matrix
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :param occlusion_tol: float used for comparing 3d points calculated from depth buffer vs input 3d points (optional)
    :return: numpy array (2, n) where n is the number of points
    """

    # reproject 2d point to 3d using depth image
    pts_3d_reproj = project_2d_to_3d(pts_2d, depth, K, pose, w, h)
    err = pts_3d - pts_3d_reproj

    T_m_c = pose2matrix(pose)
    T_c_m = T_inv(T_m_c)

    # vector homogenous coordinates, use zeros instead of ones
    err1 = np.vstack([err, np.zeros((1, err.shape[1]))])
    err_c = T_c_m.dot(err1)
    err_z = np.absolute(err_c[2, :]).reshape(-1)

    # only keep points if Z buffer is within tolerance value
    pts_2d = pts_2d[:, err_z <= occlusion_tol]

    return pts_2d


def project_2d_to_3d(pts_2d, D, K, pose, w, h, return_valid_ind=False):
    """
    Takes 2d points (pts_2d) in image coordinates from inspection images and projects them to world coordinates.
    :param pts_2d: numpy array (2, n) where n is the number of points
    :param D: depth image
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :param K: camera matrix
    :return: numpy array (3, n) where n is the number of points
    """

    if not check_visible(pts_2d, w, h):
        null_arr = np.empty((3, pts_2d.shape[1]))
        null_arr[:, :] = None
        if return_valid_ind:
            return null_arr, []
        else:
            return null_arr        

    sx = D.shape[1] / w
    sy = D.shape[0] / h

    u = pts_2d[0, :] * sx
    v = pts_2d[1, :] * sy

    fx = K[0, 0] * sx
    fy = K[1, 1] * sy
    cx = K[0, 2] * sx
    cy = K[1, 2] * sy

    d = D[tuple(np.array(v, dtype=int)), tuple(np.array(u, dtype=int))]
    valid_mask = d > 0
    u = u[valid_mask]
    v = v[valid_mask]
    d = d[valid_mask]
    d = d.reshape(-1)

    z_c = np.array(d, dtype=np.float) / 1000
    x_c = z_c * (u - cx) / fx
    y_c = z_c * (v - cy) / fy

    pts3d_c = np.array([x_c, y_c, z_c, np.ones(x_c.shape[0])])

    tc = pose[:3]
    qc = pose[3:]

    T_m_c = np.eye(4)
    T_m_c[:3, :3] = Rotation.from_quat(qc).as_matrix()
    T_m_c[:3, 3] = tc

    pts_3d = T_m_c.dot(pts3d_c)
    pts_3d = pts_3d[:3, :] / pts_3d[3, :]

    if return_valid_ind:
        return pts_3d, np.where(valid_mask)[0]
    else:
        return pts_3d


def pose2matrix(pose):
    """
    Takes camera pose and calculates transformation matrix from camera coordinates to world coordinates
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :return: numpy array transformation matrix (4x4)
    """

    p = pose[:3]
    q = pose[3:]
    R = Rotation.from_quat(q)
    T_m_c = np.eye(4)
    T_m_c[:3, :3] = R.as_matrix()
    T_m_c[:3, 3] = p
    return T_m_c

def matrix2pose(T_m_c):
    R = T_m_c[:3,:3]
    p = T_m_c[:3,3]
    q = Rotation.from_matrix(R).as_quat()

    pose = np.concatenate([p,q])

    return pose

def euler2quat(rot):
    R = Rotation.from_euler("zyx", rot)
    return R.as_quat()


def pose2vecs(pose):
    """
    Takes camera pose in world coordinates and calculates translation and rotation vectors in camera coordinates
    :param pose: tuple representing pose of camera (tx, ty, tz, qx, qy, qz, qw)
    :return: two numpy arrays (3,)
    """

    T_m_c = pose2matrix(pose)

    R = T_m_c[:3, :3]
    C = T_m_c[:3, 3]

    rvec = cv2.Rodrigues(R.T)[0]
    tvec = -R.T.dot(C.reshape(3, 1))

    return tvec, rvec


def T_inv(Tmat):
    """
    Calculates exact solution of inverse of transformation matrix for maximum speed & precsion
    :param Tmat: np array transformation matrix (4x4)
    :return: np array inverse transformation matrix (4x4)
    """
    R = Tmat[:3, :3]
    t = Tmat[:3, 3]
    Tmat_inv = np.eye(4)
    Tmat_inv[:3, :3] = R.T
    Tmat_inv[:3, 3] = -R.T.dot(t)
    return Tmat_inv


def arr_to_list_of_tuples(arr):
    """
    Converts array of 2D or 3D points into a list of tuples
    :param arr: array of 2D points (2,n) or 3D points (3,n)
    :return: list of tuples
    """
    return list(map(tuple, arr.T))


def check_visible(pts_2d, w, h):
    """
    Check if all 2D point are visible in image
    :param pts_2d: numpy array (2, n) where n is the number of points
    :param w: int width of image
    :param h: int height of image
    :return bool: return True if all 2D points are visible
    """
    if pts_2d.shape[0] == 0:
        return False

    visible_bools = np.vstack(
        [pts_2d[0, :] >= 0, pts_2d[0, :] < w, pts_2d[1, :] >= 0, pts_2d[1, :] < h]
    )
    visible_bool = np.all(visible_bools.reshape(-1))
    return visible_bool


def fit_bb_2d(im_bw):
    """
    Fits a bounding box to a binary image
    :param im_bw: binary image
    :return: x1, y1, x2, y2, cx, cy, dx, dy, h, w
    """

    # fit bounding xyxy form
    true_pxls = np.argwhere(im_bw)
    # get the height and width of the image
    (h, w) = im_bw.shape[:2]
    # calculate the bounding box in xyxy
    x1, y1, x2, y2 = (
        true_pxls[:, 1].min(),
        true_pxls[:, 0].min(),
        true_pxls[:, 1].max(),
        true_pxls[:, 0].max(),
    )
    # calculate the delta in x and y in pixel units
    dx = x2 - x1
    dy = y2 - y1
    # calculate the centroid the bounding box
    cx = x1 + dx / 2
    cy = y1 + dy / 2
    # return lots of values
    return x1, y1, x2, y2, cx, cy, dx, dy, h, w


def find_homography(im):
    """
    Takes the outline of the projected plan image and finds a homography that
    maximizes the resolution in the image
    :param im_gray:
    :return: the homography matrix and the ccw rotation angle
    """
    # convert grayscale image to binary
    if im.ndim == 3:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = im
    (thresh, im_bw) = cv2.threshold(
        im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # get initial bounding box
    x1, y1, x2, y2, cx, cy, dx, dy, h, w = fit_bb_2d(im_bw)
    # save the best size of the bounding bbox (minimize this value)
    best_size = dx * dy
    # save the rotation matrix (should be zero rotational matrix)
    best_M = cv2.getRotationMatrix2D((cx, cy), 0, 1)
    x1_best, y1_best, x2_best, y2_best = x1, y1, x2, y2
    angle_i = 0

    # loop through rotations (a minimum should be achieved within 90 degrees)
    # there might be an early stop condition... (i.e., when i > 45, dx*dy > best_size then break)
    # rotations are CCW

    # Initialize dx_best dy_best
    M = cv2.getRotationMatrix2D((cx, cy), 0, 1)
    im_rot = cv2.warpAffine(im_bw, M, (w, h))
    x1_best, y1_best, x2_best, y2_best, _, _, dx_best, dy_best, _, _ = fit_bb_2d(
        im_rot
    )

    for i in range(0, 90):
        M = cv2.getRotationMatrix2D((cx, cy), i, 1)
        im_rot = cv2.warpAffine(im_bw, M, (w, h))
        x1, y1, x2, y2, _, _, dx, dy, _, _ = fit_bb_2d(im_rot)
        if dx * dy < best_size:
            best_size = dx * dy
            best_M = M
            x1_best, y1_best, x2_best, y2_best, dx_best, dy_best = (
                x1,
                y1,
                x2,
                y2,
                dx,
                dy,
            )
            angle_i = i
        elif i > 45 and dx * dy > best_size:
            break
        else:
            continue

    # however if dy > dx this means the bounding box is in portrait form
    # so we need to rotate it by 90 degrees ccw more so that dx > dy
    if dy > dx:
        angle_i = angle_i + 90
        best_M = cv2.getRotationMatrix2D((cx, cy), angle_i, 1)
        im_rot = cv2.warpAffine(im_bw, best_M, (w, h))
        x1_best, y1_best, x2_best, y2_best, _, _, dx_best, dy_best, _, _ = fit_bb_2d(
            im_rot
        )

    M_inv = cv2.getRotationMatrix2D((cx, cy), -angle_i, 1)
    points = np.array(
        [
            [x1_best, y1_best],
            [x1_best, y2_best],
            [x2_best, y1_best],
            [x2_best, y2_best],
        ]
    )
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    transformed_points = M_inv.dot(points_ones.T).T

    # for point in transformed_points:
    #    im_bw = cv2.circle(im_bw, (int(point[0]), int(point[1])), radius=5, color=(255, 0, 0), thickness=3)

    # optimized for landscape view
    scale_factor = min(w / dx_best, h / dy_best)
    y_buffer = (h - (dy_best * scale_factor)) / 2
    x_buffer = (w - (dx_best * scale_factor)) / 2
    points_dest = np.array(
        [
            [int(x_buffer), int(y_buffer)],
            [int(x_buffer), int(h - y_buffer)],
            [int(w - x_buffer), int(y_buffer)],
            [int(w - x_buffer), int(h - y_buffer)],
        ]
    )

    H, mask = cv2.findHomography(transformed_points, points_dest)
    return H, angle_i


def calculate_area(pts_3d):
    """
    Calculate the area of a planar polygon
    :param pts_3d: 3d points (3xn) of the polygon
    :return: the area of the polygon
    """
    # fit plane through first 3 points
    plane = calc_gnd_plane(arr_to_list_of_tuples(pts_3d[:, :3]))
    # find the transformation matrix from the world to plane
    T_p_m = find_plane_transformation(plane)
    # projection matrix with K = Identity so 2d points are in real scale
    P = T_p_m[:3, :]
    # convert 3d points to 2d plane coordinates
    pts_2d = world_pt_to_plan_img(
        pts_3d, P, return_ints=False, return_tuples=True, normalize_w=False
    )
    # create Polygon object
    polygon = Polygon(pts_2d)
    # calculate area
    return np.abs(polygon.area)


# def slice_median_filter(pointcloud, gnd_plane, hmin, hmax, intervals=10):
#     I_l = []
#     for h in np.linspace(hmin,hmax,intervals):
#         pc, T_p_m = slice_pc(pointcloud, gnd_plane, height=h)
#         if len(pc.points)>0:
#             I, P = create_plan_view(pc, T_p_m)
#             I_l.append(I)

#     I = np.median(I_l,axis=0)
#     I[I<I.max()*0.5] = 0
#     return I,P


if __name__ == "__main__":
    gnd_points = load_gnd_pts(
        os.path.join(
            Path(os.getcwd()).resolve().parents[0], "sample_data", "picking_list.txt"
        )
    )
    pointcloud = load_pc(
        os.path.join(Path(os.getcwd()).resolve().parents[0], "sample_data", "cloud.ply")
    )
    gnd_plane = calc_gnd_plane(gnd_points)
    pc, T_p_m = slice_pc(pointcloud, gnd_plane, height=2.5)
    I, P = create_plan_view(pc, T_p_m)

    # Test points
    pts_3d = np.array(
        [
            [3.616041, 1.146156, 2.121655],
            [3.616041, 1.146156, 2.121655],
            [3.616041, 1.146156, 2.121655],
            [3.616041, 1.146156, 2.121655],
        ]
    )
    pts_3d = pts_3d.transpose()  # send as columns

    img_coords = world_pt_to_plan_img(pts_3d, P, return_tuples=True)

    # save test outputs
    # import cv2
    # cv2.imwrite('test.jpg',I)
    # o3d.io.write_point_cloud('test.pcd',pc)
