import cv2
import numpy as np

def intrinsics_resize(camera_matrix, image_size, new_image_size):
    """
    Resize the intrinsics matrix for a new image size.

    Parameters
    ----------
    camera_matrix : numpy.ndarray
        The 3x3 camera matrix.
    image_size : tuple
        The original image size (width, height).
    new_image_size : tuple
        The new image size (width, height).

    Returns
    -------
    new_camera_matrix : numpy.ndarray
        The resized 3x3 camera matrix.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    width_ratio = new_image_size[0] / image_size[0]
    height_ratio = new_image_size[1] / image_size[1]

    new_fx = fx * width_ratio
    new_fy = fy * height_ratio
    new_cx = cx * width_ratio
    new_cy = cy * height_ratio

    new_camera_matrix = np.array([
        [new_fx, 0.0, new_cx],
        [0.0, new_fy, new_cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return new_camera_matrix

def estimate_camera_parameters(
    pts_3d,
    pts_2d,
    image_size,
    coarse_intrinsics=None,
    known_intrinsics=False,
    estimate_distortion=False
):
    """
    Estimate camera intrinsics and extrinsics from 2D-3D correspondences.

    Parameters
    ----------
    pts_3d : numpy.ndarray
        Array of shape (N, 3) with the 3D points in the world coordinate system.
    pts_2d : numpy.ndarray
        Array of shape (N, 2) with the corresponding 2D pixel coordinates.
    image_size : tuple
        (width, height) of the image, e.g. (1920, 1080).
    known_intrinsics : numpy.ndarray or None
        If None, the intrinsics will be estimated.
        If not None, must be a (3, 3) matrix specifying fx, fy, cx, cy, etc.
    estimate_distortion : bool
        If True, also estimate lens distortion coefficients (k1, k2, p1, p2...).
        If False, set them to zero.

    Returns
    -------
    camera_matrix : numpy.ndarray
        The 3x3 intrinsic matrix.
    dist_coeffs : numpy.ndarray
        The distortion coefficients (if estimated; otherwise zero).
    rvec : numpy.ndarray
        The rotation vector (Rodrigues) that transforms 3D world coords to the camera frame.
    tvec : numpy.ndarray
        The translation vector that transforms 3D world coords to the camera frame.

    Notes
    -----
    - If `known_intrinsics` is provided, we only estimate rvec, tvec via solvePnP.
    - If `known_intrinsics` is None, we use calibrateCamera to solve for everything.
    """

    # Convert inputs to the correct float32 shape for OpenCV
    pts_3d = np.asarray(pts_3d, dtype=np.float32).reshape(-1, 3)
    pts_2d = np.asarray(pts_2d, dtype=np.float32).reshape(-1, 2)

    if known_intrinsics:
        # ------------------------------------------------------------
        # Path 1: Known Intrinsics => Solve only for extrinsics
        # ------------------------------------------------------------
        camera_matrix = coarse_intrinsics.copy()
        # Typically we assume no distortion if not explicitly given
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Ensure 3D object points have shape (N,1,3)
        pts_3d = np.array(pts_3d, dtype=np.float32).reshape(-1, 1, 3)
        # Ensure 2D image points have shape (N,1,2)
        pts_2d = np.array(pts_2d, dtype=np.float32).reshape(-1, 1, 2)

        # Use solvePnP to find the rotation/translation
        # flags can be SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_AP3P, etc.
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise RuntimeError("solvePnP failed to find a solution.")

    else:
        # ------------------------------------------------------------
        # Path 2: Unknown Intrinsics => calibrateCamera
        # ------------------------------------------------------------
        # calibrateCamera expects arrays-of-arrays for each view,
        # even if we treat all correspondences as a single "view".
        # Ensure 3D object points have shape (N,1,3)
        pts_3d = np.array(pts_3d, dtype=np.float32).reshape(-1, 1, 3)
        # Ensure 2D image points have shape (N,1,2)
        pts_2d = np.array(pts_2d, dtype=np.float32).reshape(-1, 1, 2)

        # Wrap them in lists (each image should have one array of (N,1,3) and (N,1,2))
        object_points_list = [pts_3d]
        image_points_list = [pts_2d]
        print("Calibrating camera intrinsics...")


        # Initialize guess for camera matrix or let calibrateCamera do it
        # If you do have some approximate guess, you can pass it here
        init_camera_matrix = np.array(coarse_intrinsics, dtype=np.float32)
        dist_coeffs_init = None

        # Termination criteria for corner refinement, etc. (not crucial here, but good to have)
        # ret = reprojection error
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_list,
            imagePoints=image_points_list,
            imageSize=image_size,
            cameraMatrix=init_camera_matrix,
            distCoeffs=dist_coeffs_init,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )

        # If we didn't want to estimate distortion at all, set dist_coeffs to zero
        if not estimate_distortion:
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # For a single "view", calibrateCamera returns a list with one rvec/tvec
        rvec = rvecs[0]
        tvec = tvecs[0]

    R = cv2.Rodrigues(rvec)[0]  # convert rotation vector to rotation matrix

    return camera_matrix, dist_coeffs, R, tvec

def proj_3d_to_2d(points_3d, R, t, K):
    """
    Projects 3D points to 2D using the pinhole model: x = K [R|t] X
    Ignores lens distortion.

    Parameters
    ----------
    points_3d : np.ndarray
        Shape (N, 3) array of 3D points in the same coordinate system as R, t.
    R : np.ndarray
        3x3 rotation matrix.
    t : np.ndarray
        3x1 translation vector (or shape (3,) is also okay).
    K : np.ndarray
        3x3 intrinsic camera matrix.

    Returns
    -------
    points_2d : np.ndarray
        Shape (N, 2) array of 2D points (pixel coordinates).
    """
    # Ensure correct shapes
    points_3d = np.asarray(points_3d, dtype=np.float32)
    if points_3d.ndim == 1 and points_3d.size == 3:
        points_3d = points_3d.reshape(1, 3)  # single point
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3, 1)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)

    # 1) Transform points from world to camera coordinates: X_cam = R * X_world + t
    #    We'll do this in a batch using matrix multiplication.
    #    points_3d is (N,3) -> transpose to (3,N), multiply, then transpose back.
    X_cam = R @ points_3d.T + t  # shape (3, N)
    
    # 2) Project onto image plane with intrinsics: x_img = K * X_cam
    x_img = K @ X_cam  # shape (3, N)

    # 3) Convert from homogeneous to (u,v): (u, v) = (x_img[0]/x_img[2], x_img[1]/x_img[2])
    #    We'll handle possible division by zero if a point has z=0 or negative depth.
    z = x_img[2, :]
    # Avoid division by zero (or negative depth which is behind the camera)
    valid = z > 1e-6  # you can set your own threshold

    u = x_img[0, valid] / z[valid]
    v = x_img[1, valid] / z[valid]

    # Return Nx2 array of 2D pixels
    points_2d = np.column_stack((u, v))
    return points_2d



# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we have N=20 correspondences from multiple frames or one frame
    # 3D world coords (e.g., the robot end-effector tip positions)
    pts_3d = np.load('ee_traj_6d.npy')[..., :3]  # shape (N, 3)
    # Corresponding 2D image pixels
    pts_2d = np.load('ee_traj_2d.npy')  # shape (N, 2)

    # Your image resolution, e.g. 1920x1080
    image_width = 256
    image_height = 256

    coarse_intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ) # logitech C920
    coarse_intrinsic = intrinsics_resize(coarse_intrinsic, (640, 480), (256, 256))

    # Case A: Unknown intrinsics
    cam_mtx, dist, rvec, tvec = estimate_camera_parameters(
        pts_3d, pts_2d,
        (image_width, image_height),
        coarse_intrinsics=coarse_intrinsic,
        known_intrinsics=True,
        estimate_distortion=False
    )
    print("=== Unknown Intrinsics ===")
    print("Camera Matrix:\n", cam_mtx)
    print("Distortion:\n", dist.ravel())
    print("Rotation Vector:\n", rvec.ravel())
    print("Translation Vector:\n", tvec.ravel())

    # # Case B: Known intrinsics (assume some nominal camera matrix, e.g., from spec)
    # known_camera_matrix = np.array([
    #     [1500.,   0.0,  960.],  # fx=1500, cx=960
    #     [0.0,   1500.,  540.],  # fy=1500, cy=540
    #     [0.0,     0.0,    1.0]
    # ], dtype=np.float32)

    # cam_mtx2, dist2, rvec2, tvec2 = estimate_camera_parameters(
    #     pts_3d, pts_2d,
    #     (image_width, image_height),
    #     known_intrinsics=known_camera_matrix,
    #     estimate_distortion=False
    # )
    # print("\n=== Known Intrinsics ===")
    # print("Camera Matrix:\n", cam_mtx2)
    # print("Distortion:\n", dist2.ravel())  # should remain zero
    # print("Rotation Vector:\n", rvec2.ravel())
    # print("Translation Vector:\n", tvec2.ravel())