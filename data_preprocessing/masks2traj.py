import numpy as np
import cv2
from scipy.ndimage import label, center_of_mass
from scipy.interpolate import interp1d
from PIL import Image
def preprocess_mask(mask, min_size=50):
    """Remove small holes and filter out small connected components.
    min_size: minimum size of connected components to keep, w.r.t. a 256x256 image.
    """
    min_size = min_size * (mask.shape[-1] * mask.shape[-2]) // 256**2
    large_size_threshold = 4 * min_size
    # Close small holes
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # debug only
    # (Image.fromarray(mask.astype(np.uint8)*255).convert('L')).save('ori_mask.png')
    # (Image.fromarray(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))*255)).save('2_close_hole_mask.png')

    # Label connected components
    labeled_mask, num_features = label(mask)
    sizes = np.bincount(labeled_mask.ravel())

    if np.any(sizes[1:] >= large_size_threshold):  # Ignore background (index 0)
        min_size = sizes[1:].max() // 3  # Increase min_size threshold

    if (sizes[1:]>2*min_size).sum() >= 2: # if there are more than 2 large objects, keep only the 2 largest
        min_size = np.sort(sizes[1:])[-2]

    mask_filtered = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        if sizes[i] >= min_size:
            mask_filtered[labeled_mask == i] = 1
    
    return mask_filtered

def get_centroid(mask, mode='bbox'): # choose between 'bbox' and 'mass'
    """Compute the centroid of a binary mask using bounding box center."""
    if np.sum(mask) == 0:
        return None
    if mode == 'bbox':
        mask_indices = np.argwhere(mask > 0)
        min_y, min_x = mask_indices.min(axis=0)
        max_y, max_x = mask_indices.max(axis=0)
        return (min_x + max_x) // 2, (min_y + max_y) // 2
    elif mode == 'mass':
        return center_of_mass(mask)

def compute_trajectory(masks, max_dist=40):
    """Compute a single, consistent trajectory from a sequence of binary masks.
    max_dist: maximum distance between consecutive frames to be considered as valid. w.r.t. a 256x256 image.
    """
    max_dist = max_dist * masks.shape[-1] // 256
    centers = []
    modified_masks = []
    for mask in masks:
        mask = preprocess_mask(mask)
        centroid = get_centroid(mask)
        centers.append(centroid if centroid else (None, None))
        modified_masks.append(mask)
    
    # Convert to numpy for easier processing
    centers = np.array(centers, dtype=float)
    
    # Reject outliers (masks that jump too far)
    for i in range(1, len(centers)):
        if np.linalg.norm(centers[i] - centers[i - 1]) > max_dist:
            centers[i] = np.nan

    # Fill missing values with interpolation
    valid_idx = np.where(~np.isnan(centers[:, 0]))[0]
    invalid_idx = np.where(np.isnan(centers[:, 0]))[0]
    
    if len(invalid_idx) > 0 and len(valid_idx) > 1:
        interp_x = interp1d(valid_idx, centers[valid_idx, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(valid_idx, centers[valid_idx, 1], kind='linear', fill_value='extrapolate')
        centers[invalid_idx, 0] = interp_x(invalid_idx)
        centers[invalid_idx, 1] = interp_y(invalid_idx)

    return centers, np.stack(modified_masks, axis=0)
