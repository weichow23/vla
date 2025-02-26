import tensorflow_datasets as tfds
import tensorflow as tf
import dlimp as dl
import os
import json
import numpy as np
from dataloader_debug.tmp import traj_transforms, transforms
from tqdm import tqdm

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

dataset_name = "bridge_orig"
data_path = '/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/OXE'
num_parallel_calls = tf.data.AUTOTUNE


metadata_path = os.path.join(data_path, dataset_name, "camera_calibration.json")
assert os.path.isfile(metadata_path), \
    f"use_cam_pose or use_ego_action is set but cam_pose_path is not found in {metadata_path}"
camera_calib = json.load(open(metadata_path, "r"))
camera_calib = {k: v["0"] for k, v in camera_calib.items()} # might be none in rare cases
camera_calib = {
    k: traj_transforms.t_euler_to_transformation_matrix(np.concatenate([np.array(v["t"]), np.array(v["r"])])) \
        for k, v in camera_calib.items() if v is not None}
camera_keys = tf.constant(list(camera_calib.keys()), dtype=tf.string)
camera_values = tf.cast(tf.stack(list(camera_calib.values()), axis=0), dtype=tf.float32)  # Stack to ensure consistent shape
camera_values = tf.reshape(camera_values, [camera_values.shape[0], 16])  # Flatten each 4x4 matrix

camera_table = tf.lookup.experimental.DenseHashTable(
    key_dtype=tf.string,
    value_dtype=tf.float32,
    default_value=tf.zeros([16], dtype=tf.float32),
    empty_key="",
    deleted_key="<deleted>"
)

camera_table.insert(camera_keys, camera_values)

image_obs_keys = {"primary": "image_0"}
language_key = "language_instruction"
state_obs_keys = ["state"]
absolute_action_mask = np.array([False, False, False, False, False, False, True])
absolute_action_mask = tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)

standardize_fn = transforms.bridge_orig_dataset_transform

def has_valid_cam_pose(traj):
    """
    Returns True if the trajectory has a valid camera pose in the lookup table.
    """
    episode_file = traj['traj_metadata']['episode_metadata']['file_path'][0]
    episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"][0]
    key = tf.strings.join([episode_file, tf.strings.as_string(episode_id)], separator="~")
    return tf.reduce_any(camera_table.lookup(key) != 0)

def get_cam_pose(traj):
    episode_file = traj['traj_metadata']['episode_metadata']['file_path'][0]
    episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"][0]
    
    key = tf.strings.join([episode_file, tf.strings.as_string(episode_id)], separator="~")
    traj_len = tf.shape(traj["action"])[0]

    calib = camera_table.lookup(key)
    calib = tf.reshape(calib, [4, 4])
    calib_tiled = tf.tile(
        tf.expand_dims(calib, axis=0),
        [traj_len, 1, 1]
    )

    traj["observation"]["camera_calib"] = calib_tiled
    return traj

def restructure(traj):
    # apply a standardization function, if provided
    if standardize_fn is not None:
        traj = standardize_fn(traj)

    # extracts images, depth images and proprio from the "observation" dict
    traj_len = tf.shape(traj["action"])[0]
    old_obs = traj["observation"]
    new_obs = {}
    for new, old in image_obs_keys.items():
        if old is None:
            new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
        else:
            new_obs[f"image_{new}"] = old_obs[old]

    if state_obs_keys:
        new_obs["proprio"] = tf.concat(
            [
                (
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                )
                for key in state_obs_keys
            ],
            axis=1,
        )

    if "camera_calib" in old_obs:
        new_obs["camera_calib"] = old_obs["camera_calib"]

    # add timestep info
    new_obs["timestep"] = tf.range(traj_len)

    # extracts `language_key` into the "task" dict
    task = {}
    if language_key is not None:
        if traj[language_key].dtype != tf.string:
            raise ValueError(
                f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
            )
        task["language_instruction"] = traj.pop(language_key)

    traj = {
        "observation": new_obs,
        "task": task,
        "action": tf.cast(traj["action"], tf.float32),
        "dataset_name": tf.repeat(dataset_name, traj_len),
    }        

    if absolute_action_mask is not None:
        # Convert mask to tensor
        abs_mask = tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)
        mask_len = tf.shape(abs_mask)[0]
        action_dim = tf.shape(traj['action'])[-1]
        
        # Use tf.debugging.assert_equal instead of direct comparison
        tf.debugging.assert_equal(
            mask_len,
            action_dim,
            message=f"Length of absolute_action_mask and action dimension must match"
        )
        
        traj["absolute_action_mask"] = tf.tile(
            tf.expand_dims(abs_mask, axis=0),
            [traj_len, 1],
        )

    return traj


builder = tfds.builder(dataset_name, data_dir=data_path)
dataset = dl.DLataset.from_rlds(builder, split='all', shuffle=False, num_parallel_reads=num_parallel_calls)

dataset = dataset.filter(has_valid_cam_pose)
dataset = dataset.traj_map(get_cam_pose, num_parallel_calls)
dataset = dataset.traj_map(traj_transforms.transform_poses_to_camera, num_parallel_calls)
dataset = dataset.traj_map(restructure, num_parallel_calls)
for traj in tqdm(dataset):
    pass
