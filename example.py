import cv2
import numpy as np

def calibrate_single_view(object_points_3d, image_points_2d, image_size):
    """
    Calibrate a camera (intrinsics + extrinsics) given 2D-3D correspondences for a single view.

    Parameters
    ----------
    object_points_3d : numpy.ndarray
        Array of shape (N, 3) containing N known 3D points in the world.
    image_points_2d : numpy.ndarray
        Array of shape (N, 2) containing the corresponding N 2D points in the image.
    image_size : tuple
        (width, height) of the image in pixels.

    Returns
    -------
    camera_matrix : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    dist_coeffs : numpy.ndarray
        Distortion coefficients (k1, k2, p1, p2, k3, ...) depending on the model.
    rvec : numpy.ndarray
        The rotation vector for this view.
    tvec : numpy.ndarray
        The translation vector for this view.
    """

    # OpenCV calibrateCamera requires data as lists of arrays of points
    # even for one view, so we put them inside a list.
    object_points = [object_points_3d.astype(np.float32)]
    image_points  = [image_points_2d.astype(np.float32)]

    # Initial guess for camera matrix can be set to None; OpenCV will initialize internally.
    # Distortion coefficients also set to None initially.
    flags = 0
    ret_val, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )

    # calibrateCamera returns:
    #   - ret_val: the RMS re-projection error
    #   - camera_matrix: the 3x3 intrinsic matrix
    #   - dist_coeffs: distortion coefficients
    #   - rvecs, tvecs: rotation and translation vectors for each view

    # For a single view, we only have rvecs[0] and tvecs[0].
    return camera_matrix, dist_coeffs, rvecs[0], tvecs[0]


# ----------------- EXAMPLE USAGE ----------------- #

if __name__ == "__main__":
    # Suppose we know 4 or more 3D points in the world:
    # Example: corners of a known marker or object in 3D
    object_points_3d = np.array([
        [0.0,  0.0,  0.0],
        [10.0, 0.0,  0.0],
        [10.0, 10.0, 0.0],
        [0.0,  10.0, 0.0]
    ])

    # And the corresponding points in the image (in pixel coordinates)
    image_points_2d = np.array([
        [250.0, 250.0],
        [350.0, 250.0],
        [350.0, 350.0],
        [250.0, 350.0]
    ])

    # Your image resolution (width, height) in pixels:
    image_size = (640, 480)

    # Calibrate the camera using the single view
    camera_matrix, dist_coeffs, rvec, tvec = calibrate_single_view(
        object_points_3d,
        image_points_2d,
        image_size
    )

    print("Camera Matrix (Intrinsics):\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    print("\nRotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)

    # Optional: Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    print("\nRotation Matrix:\n", R)

    # Form the 3x4 Extrinsic matrix: [R | t]
    extrinsic_matrix = np.hstack((R, tvec.reshape(-1, 1)))
    print("\nExtrinsic Matrix:\n", extrinsic_matrix)