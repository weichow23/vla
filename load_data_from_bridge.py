"""
1. create a json file that contains the task for each episode
2. create a json file that contains the episodes with instructions to use spoon and towel -- which hopefully are the same tasks
"""
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import h5py
import time
from contextlib import contextmanager
from PIL import Image

@contextmanager
def retry_h5py_file(file_path, mode='a', retries=2, delay=1, verbose=True):
    """
    Context manager for opening an HDF5 file with retry logic.
    
    Args:
        file_path (str): Path to the HDF5 file.
        mode (str): Mode to open the file ('r', 'r+', 'a', etc.).
        retries (int): Number of retry attempts.
        delay (int): Delay (in seconds) between retries.

    Yields:
        h5py.File: Opened HDF5 file object.
    """
    for attempt in range(retries):
        try:
            with h5py.File(file_path, mode) as h5file:
                yield h5file
                return  # Exit the loop once successful
        except BlockingIOError as e:
            if attempt < retries - 1:
                if verbose:
                    print(f"Failed to open file '{file_path}' (attempt {attempt + 1}/{retries}). Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                raise e  # Re-raise after exhausting retries

def load_all_data(hdf5_path, keys=None, cameras=None, steps=None, temporal_window=21, BGR2RGB=True):
    """
    Load specified data from the HDF5 file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        keys (list, optional): List of keys to load. Valid options: 
            'pose', 'intrinsics', 'depth', 'dyn_masks', 'tracks', 'rgb', 'action'.
        cameras (list, optional): List of cameras to load data for.
        steps (list, optional): List of steps to load data for.
        temporal_window (int, optional): Temporal window for track queries. Default is 21.
        BGR2RGB (bool, optional): Whether to convert RGB data from BGR to RGB. Default is True.
    
    Returns:
        dict: A dictionary with keys as requested data and corresponding loaded values.
    """
    if keys is None:
        keys = ['pose', 'intrinsics', 'depth', 'dyn_masks', 'tracks', 'rgb', 'action', 'task', 'text_embeds']
    reversed_indices = None
    if steps is not None:
        original_steps = steps
        unique_sorted_list = sorted(set(original_steps))
        reversed_indices = np.array([unique_sorted_list.index(value) for value in original_steps])
        steps = np.array(unique_sorted_list)

    out_data = {}
    with retry_h5py_file(hdf5_path, 'r') as h5file:
        # Load requested data
        if 'pose' in keys and 'pose' in h5file:
            poses = h5file["pose"]
            if cameras:
                poses = {cam: np.array(poses[cam][steps][reversed_indices]) if steps is not None else np.array(poses[cam]) for cam in cameras}
            elif steps is not None:
                poses = {cam: np.array(poses[cam][steps][reversed_indices]) for cam in poses.keys()}
            else:
                poses = {cam: np.array(poses[cam]) for cam in poses.keys()}
            out_data['pose'] = poses
        
        if 'intrinsics' in keys and 'intrinsics' in h5file:
            intrinsics = h5file["intrinsics"]
            out_data['intrinsics'] = {cam: np.array(intrinsics[cam]) for cam in cameras} if cameras else {
                cam: np.array(intrinsics[cam]) for cam in intrinsics.keys()}
        
        if 'depth' in keys and 'depth' in h5file:
            depth = h5file["depth"]
            if cameras:
                depth = {cam: np.array(depth[cam][steps][reversed_indices]) if steps is not None else np.array(depth[cam]) for cam in cameras}
            elif steps is not None:
                depth = {cam: np.array(depth[cam][steps][reversed_indices]) for cam in depth.keys()}
            else:
                depth = {cam: np.array(depth[cam]) for cam in depth.keys()}
            out_data['depth'] = depth
        
        if 'action' in keys and 'action' in h5file: # 20 in total, 3 pos1 + 6 rot1 + 3 pos2 + 6 rot2 + 1 gripper1 + 1 gripper2
            action = h5file["action"]
            if steps is not None:
                action = action[steps][reversed_indices]
            out_data['action'] = np.array(action)

        if 'dyn_masks' in keys and 'dyn_masks' in h5file:
            dyn_masks = h5file["dyn_masks"]
            if cameras:
                dyn_masks = {cam: np.array(dyn_masks[cam][steps][reversed_indices]) if steps is not None else np.array(dyn_masks[cam]) for cam in cameras}
            elif steps is not None:
                dyn_masks = {cam: np.array(dyn_masks[cam][steps][reversed_indices]) for cam in dyn_masks.keys()}
            else:
                dyn_masks = {cam: np.array(dyn_masks[cam]) for cam in dyn_masks.keys()}
            out_data['dyn_masks'] = dyn_masks

        if 'tracks' in keys and 'tracks' in h5file:
            tracks = {}
            track_cams = cameras if cameras else h5file["tracks"].keys()
            for cam in track_cams:
                if cam in h5file["tracks"]:
                    data = h5file["tracks"][cam]
                    if 'tracking_setting' in data.attrs:
                        temporal_window = int(dict(item.split('=') for item in data.attrs['tracking_setting'].split(', '))['temporal_window'])
                    padding = data.attrs["padding"]
                    out_steps = []
                    query_ind = []
                    if steps is None:
                        loop_steps = np.arange(len(data))
                    else:
                        loop_steps = steps[reversed_indices]
                    for step in loop_steps:
                        dyn_pts = np.array(data[f"timestep_{step}"])
                        if dyn_pts.shape[0] == 0:
                            out_steps.append(None)
                            query_ind.append(None)
                        else:
                            out_steps.append(dyn_pts)
                            query_ind.append(temporal_window - padding[step, 0] - 1)
                    tracks[cam] = (out_steps, query_ind)
                else:
                    tracks[cam] = ([], [])
            out_data['tracks'] = tracks
        # (rgb['scene_left'][np.sort(steps)][reversed_indices][0]==rgb['scene_left'][3]).all()
        if 'rgb' in keys and 'rgb' in h5file:
            rgb = h5file["rgb"]
            if cameras:
                rgb_data = {cam: rgb[cam][steps][reversed_indices] if steps is not None else rgb[cam] for cam in cameras}
                rgb_data = {cam: np.array(data)[..., ::-1] if BGR2RGB else np.array(data) for cam, data in rgb_data.items()}
            elif steps is not None:
                rgb_data = {cam: np.array(rgb[cam][steps][reversed_indices])[..., ::-1] if BGR2RGB else np.array(rgb[cam][steps]) for cam in rgb.keys()}
            else:
                rgb_data = {cam: np.array(rgb[cam])[..., ::-1] if BGR2RGB else np.array(rgb[cam]) for cam in rgb.keys()}
            out_data['rgb'] = rgb_data
        if 'task' in keys and 'task' in h5file.attrs:
            out_data['task'] = h5file.attrs['task']
        if 'text_embeds' in keys and 'text_embeddings' in h5file:
            out_data['text_embeds'] = np.array(h5file['text_embeddings'])
    return out_data


data_path = '/home/gvl/Downloads/bridge_demo_set/'
episodes_metadata = json.load(open(os.path.join(data_path, 'samples_all_train.json')))

if not os.path.exists('bridge_demo_set'):
    os.makedirs('bridge_demo_set')

breakpoint()

for episode, len in episodes_metadata.items():
    episode_data = load_all_data(os.path.join(data_path, f'{episode}.h5'), keys=['dyn_masks', 'rgb'], BGR2RGB=False)
    mid_frame = len // 2
    fore_mask = episode_data['dyn_masks']['image_0'][mid_frame] # 64x64
    img = episode_data['rgb']['image_0'][mid_frame] # 256x256x3

    fore_mask = Image.fromarray(fore_mask).resize((256, 256), Image.NEAREST)
    img = Image.fromarray(img)

    save_name = episode.replace('/', '_') + '.png'

    fore_mask.save(f'bridge_demo_set/{save_name}')
    img.save(f"bridge_demo_set/{save_name.replace('.png', '_rgb.png')}")
