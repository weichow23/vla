"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

from asyncio import gather
import logging
from typing import Dict, Literal

import tensorflow as tf

import numpy as np
def t_euler_to_transformation_matrix(sixdof):
    """
    Note: This implementation uses the convention R = Rz @ Ry @ Rx.
    Adjust the order if your Euler angles follow a different convention.
    """
    translation = sixdof[:3]
    roll, pitch, yaw = sixdof[3:]

    # Rotation about x-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    
    # Rotation about y-axis (pitch)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Rotation about z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Combine rotation and translation into a single transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def euler_to_rot(euler):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    The rotation order is R = R_z * R_y * R_x.
    Supports both batched (shape [N,3]) and single (shape [3]) inputs.
    """
    euler = tf.convert_to_tensor(euler, dtype=tf.float32)
    squeeze = False
    if tf.rank(euler) == 1:
        euler = tf.expand_dims(euler, 0)
        squeeze = True

    roll  = euler[:, 0]
    pitch = euler[:, 1]
    yaw   = euler[:, 2]

    cos_r = tf.cos(roll)
    sin_r = tf.sin(roll)
    cos_p = tf.cos(pitch)
    sin_p = tf.sin(pitch)
    cos_y = tf.cos(yaw)
    sin_y = tf.sin(yaw)

    ones = tf.ones_like(roll)
    zeros = tf.zeros_like(roll)

    # Rotation about x-axis (roll)
    R_x = tf.stack([
        tf.stack([ones, zeros, zeros], axis=-1),
        tf.stack([zeros, cos_r, -sin_r], axis=-1),
        tf.stack([zeros, sin_r, cos_r], axis=-1)
    ], axis=1)
    
    # Rotation about y-axis (pitch)
    R_y = tf.stack([
        tf.stack([cos_p, zeros, sin_p], axis=-1),
        tf.stack([zeros, ones, zeros], axis=-1),
        tf.stack([-sin_p, zeros, cos_p], axis=-1)
    ], axis=1)
    
    # Rotation about z-axis (yaw)
    R_z = tf.stack([
        tf.stack([cos_y, -sin_y, zeros], axis=-1),
        tf.stack([sin_y, cos_y, zeros], axis=-1),
        tf.stack([zeros, zeros, ones], axis=-1)
    ], axis=1)
    
    # Compose the rotations: R = R_z * R_y * R_x.
    R_zy = tf.matmul(R_z, R_y)
    R = tf.matmul(R_zy, R_x)
    
    if squeeze:
        R = tf.squeeze(R, axis=0)
    return R

def rot_to_euler(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw)
    using the convention R = R_z * R_y * R_x.
    Supports both batched (shape [N,3,3]) and single (shape [3,3]) inputs.
    """
    R = tf.convert_to_tensor(R, dtype=tf.float32)
    squeeze = False
    if tf.rank(R) == 2:
        R = tf.expand_dims(R, 0)
        squeeze = True

    # From the rotation matrix we have:
    #   R[2,0] = -sin(pitch)  =>  pitch = asin(-R[2,0])
    #   R[2,1] = cos(pitch)*sin(roll)
    #   R[2,2] = cos(pitch)*cos(roll)  =>  roll = atan2(R[2,1], R[2,2])
    #   R[1,0] = cos(pitch)*sin(yaw)
    #   R[0,0] = cos(pitch)*cos(yaw)   =>  yaw = atan2(R[1,0], R[0,0])
    pitch = tf.asin(-R[:, 2, 0])
    roll  = tf.atan2(R[:, 2, 1], R[:, 2, 2])
    yaw   = tf.atan2(R[:, 1, 0], R[:, 0, 0])
    
    euler = tf.stack([roll, pitch, yaw], axis=-1)
    if squeeze:
        euler = tf.squeeze(euler, axis=0)
    return euler

def transform_poses_to_camera(traj):
    """
    Transforms a set of 6D poses from the world frame to the camera frame.
    
    The input 'traj' is a dictionary with:
      - traj["camera_calib"]: a tensor of shape (6,) where the first 3 elements
        are the camera translation and the last 3 are Euler angles (roll, pitch, yaw)
        defining the world-to-camera transformation.
      - traj["observation"]["state"]: a tensor where the first 6 elements of each row
        represent a pose in the world frame (first 3 for translation and last 3 for Euler angles).
        
    The function returns a tensor of shape (N, 6) containing the transformed poses in the
    camera frame.
    """

    # Extract camera calibration (world-to-camera transform)
    calib = traj["observation"]["camera_calib"]  # shape: (N,4,4)
    t_cam = calib[..., :3, 3]     # translation: (N,3)
    R_cam = calib[..., :3, :3]         # shape: (N,3,3)
    
    # Extract the world poses: each row is [t_obj, euler_obj]
    state = traj["observation"]["state"]  # shape: (N,7)
    state_shape = tf.shape(state)
    t_obj = state[:, :3]     # object translations in world frame (N,3)
    euler_obj = state[:, 3:6] # object Euler angles (N,3)
    R_obj = euler_to_rot(euler_obj)  # shape: (N,3,3)
    
    # Transform the translation: t_new = R_cam * t_obj + t_cam.
    t_obj_expanded = tf.expand_dims(t_obj, axis=-1)  # shape: (N,3,1)
    t_new = tf.matmul(R_cam, t_obj_expanded) + tf.expand_dims(t_cam, axis=-1)
    t_new = tf.squeeze(t_new, axis=-1)  # shape: (N,3)
    
    # Transform the rotation: R_new = R_cam * R_obj.
    # Expand R_cam to (1,3,3) to broadcast over N poses.
    R_new = tf.matmul(R_cam, R_obj)         # shape: (N,3,3)
    
    # Convert the rotated matrices back to Euler angles.
    euler_new = rot_to_euler(R_new)  # shape: (N,3)
    
    remaining_state = tf.cond(
        tf.greater(state_shape[-1], 6),
        lambda: state[:, 6:],
        lambda: tf.zeros([state_shape[0], 0], dtype=state.dtype)
    )

    # Concatenate the transformed translation and rotation to form 6D poses.
    traj["observation"]["state"] = tf.concat([t_new, euler_new, remaining_state], axis=1)  # shape: (N,6) # , traj["observation"]["state"][:, 6:7]
    return traj

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["chunk_mask"] = (action_chunk_indices >= 0) & (action_chunk_indices <= goal_timestep[:, None])
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_action_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def new_chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0, left_pad: bool=True, window_sample: Literal["sliding", "range"]="sliding") -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    traj = chunk_act_obs(traj, window_size, future_action_window_size)
    left_index = 0 if left_pad else window_size - 1
    tf.assert_less(left_index, traj_len)
    def slice_first_dim(x):
        return x[left_index:]
    
    def repeat_first_dim(x):
        return tf.repeat(x, repeats=window_size, axis=0)

    traj = tf.nest.map_structure(slice_first_dim, traj)
    if window_sample == "range":
        traj = tf.nest.map_structure(repeat_first_dim, traj)
        left_range = tf.range(window_size)
        left_range_mask = ~tf.sequence_mask(left_range, window_size + future_action_window_size)
        left_range_mask = tf.tile(left_range_mask, [traj_len-left_index, 1])
        traj["chunk_mask"] = traj["chunk_mask"] & left_range_mask
        
    return traj


def chunk_as_episode(traj: Dict, frame_num: int) -> Dict:
    traj_len = tf.shape(traj['action'])[0]
    indices = tf.cast(tf.linspace(0.0, tf.cast(traj_len - 1, tf.float32), frame_num), tf.int32)
    except_keys = ['action', 'observation']
    
    def gather_element(data):
        if isinstance(data, dict):
            for key in data:
                data[key] = gather_element(data[key])
            return data
        else:
            sampled_tensor = tf.gather(data, indices, axis=0)
            return sampled_tensor

    def get_first_element(data):
        if isinstance(data, dict):
            for key in data:
                if key in except_keys:
                    continue
                data[key] = get_first_element(data[key])
            return data
        else:
            return data[-1]

    for key in except_keys:
        traj[key] = gather_element(traj[key])
    
    return get_first_element(traj)


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj

if __name__ == '__main__':
    # Create an identity calibration matrix (4x4)
    calib = tf.constant([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=tf.float32)
    
    # Create two example poses in the world frame:
    # Each pose is a 6D vector: first 3 are translation, last 3 are Euler angles.
    poses = tf.constant([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1],
                            [-1.0, -2.0, -3.0, -0.1, -0.2, -0.3, 1]], dtype=tf.float32)
    
    traj = {
        "camera_calib": calib,
        "observation": {
            "state": poses
        }
    }
    
    breakpoint()

    # Transform the poses.
    transformed_traj = transform_poses_to_camera(traj)
    
    # Since the calibration is identity, the output should match the input.
    result = transformed_traj["observation"]["state"].numpy()
    print(result)