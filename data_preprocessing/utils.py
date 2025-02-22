from contextlib import contextmanager
import h5py
import numpy as np
import time
import cv2
from PIL import Image
import imageio
import torch
import math

def save_video_with_overlay(images, classifier_output, ee_traj, save_path, reproj_traj=None, inliers=None, fps=10, idx=None):
    """
    Creates and saves a video with overlaid masks.

    Args:
        images (list of PIL.Image): List of original images.
        classifier_output (torch.Tensor): Tensor of predicted masks (batch, H, W). binarized.
        ee_traj (list of tuple): List of (x, y) tuples representing the end-effector trajectory.
        save_path (str): Path to save the video.
        fps (int): Frames per second.
    """
    assert len(images) == len(classifier_output), "Mismatch between images and masks"

    if isinstance(classifier_output, torch.Tensor):
        classifier_output = classifier_output.float().cpu().numpy()

    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', format='mp4')

    ori_size = classifier_output[0].shape[-2:]

    for i in range(len(images)):
        image = np.array(images[i])  # Convert to numpy (3, H, W)

        # Process mask
        pred_mask = classifier_output[i]
        pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))  # Convert to image
        pred_mask = pred_mask.resize((image.shape[1], image.shape[0]))  # Resize
        mask_overlay = np.array(pred_mask) / 255  # Normalize

        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[:, :, 0] = (mask_overlay * 255).astype(np.uint8)  # Red channel

        # Blend overlay with image
        blended = (0.7 * image + 0.3 * overlay).astype(np.uint8)

        if inliers is not None and i not in inliers and \
            (reproj_traj is not None and i < len(reproj_traj) and reproj_traj[i] is not None) and \
            (ee_traj is not None and i < len(ee_traj) and ee_traj[i] is not None):
            error = np.linalg.norm(ee_traj[i] - reproj_traj[i])
            cv2.putText(blended, f"outlier:{error:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw trajectory
        if i < len(ee_traj) and ~np.isnan(ee_traj[i, 0]):
            # resize the trajectory to the image size
            ee_traj[i] = (int(ee_traj[i][0] * image.shape[1] / ori_size[1]), int(ee_traj[i][1] * image.shape[0] / ori_size[0]))
            x, y = map(int, ee_traj[i])
            cv2.circle(blended, (x, y), 5, (0, 255, 0), -1)  # Green dot for trajectory

        if reproj_traj is not None and i < len(reproj_traj) and ~np.isnan(reproj_traj[i, 0]):
            # resize the trajectory to the image size
            reproj_traj[i] = (int(reproj_traj[i][0] * image.shape[1] / ori_size[1]), int(reproj_traj[i][1] * image.shape[0] / ori_size[0]))
            x, y = map(int, reproj_traj[i])
            cv2.circle(blended, (x, y), 5, (255, 0, 0), -1)

        # Add frame index
        if idx is not None:
            cv2.putText(blended, f"{idx}:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write frame to video
        writer.append_data(blended)

    writer.close()

def concat_videos_grid(video_paths, output_path, fps=None):
    """
    Concatenates multiple videos into a single grid video, looping them in reverse if they end earlier.

    Args:
        video_paths (list of str): List of paths to input videos.
        output_path (str): Path to save the concatenated video.
        fps (int, optional): Frames per second. If None, uses the FPS of the first video.
    """
    # Open video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    if any(not cap.isOpened() for cap in caps):
        raise ValueError("One or more video files could not be opened.")

    # Get frame properties from the first video
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = caps[0].get(cv2.CAP_PROP_FPS)
    fps = fps or int(original_fps or 10)  # Default to first video's FPS or 10 if unknown

    # Determine optimal grid size (rows × cols)
    num_videos = len(video_paths)
    grid_cols = math.ceil(math.sqrt(num_videos))
    grid_rows = math.ceil(num_videos / grid_cols)  # Ensure all videos fit

    # Initialize video writer
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='mp4')

    # Store frames for reversing when videos end
    stored_frames = [[] for _ in caps]
    active_flags = [True] * num_videos  # Track if a video is still reading forward

    while any(active_flags):
        frames = []
        for i, cap in enumerate(caps):
            if active_flags[i]:  # If video is still playing forward
                ret, frame = cap.read()
                if ret:
                    stored_frames[i].append(frame)  # Store frame for later reverse playback
                else:
                    active_flags[i] = False  # Mark as finished
                    stored_frames[i].reverse()  # Reverse stored frames for looping

            # Get the current frame (either from video or reversed frames)
            if not active_flags[i] and stored_frames[i]:  # Play stored frames in reverse
                frame = stored_frames[i].pop()
            elif not active_flags[i]:  # If no stored frames left, keep the last frame
                frame = stored_frames[i][-1] if stored_frames[i] else np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            frames.append(frame)

        # Fill missing slots with last available frames to complete the grid
        while len(frames) < grid_rows * grid_cols:
            frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

        # Arrange frames into a grid
        grid_frames = [frames[i:i + grid_cols] for i in range(0, len(frames), grid_cols)]
        grid_image = np.vstack([np.hstack(row) for row in grid_frames])

        # Write frame
        writer.append_data(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))

    # Release resources
    for cap in caps:
        cap.release()
    writer.close()

    print(f"Saved concatenated video to {output_path}")


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
