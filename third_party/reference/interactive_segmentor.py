
import cv2
import matplotlib
import mediapy
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline
import tensorflow_datasets as tfds
import os

sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def get_gripper_mask(img, prompt, box_or_point='box'):
    if box_or_point == 'box':
        breakpoint()
        inputs = sam_processor(img, input_boxes=[[[prompt]]], return_tensors="pt")
    elif box_or_point == 'point':
        points = [[x, y] for x, y, _ in prompt]
        points = [points]
        label = [label for _, _, label in prompt]
        label = [label]
        inputs = sam_processor(img, 
                               input_points=points, 
                               input_labels=label,
                               return_tensors="pt")

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0].numpy()

    return mask

def select_bbox_interactively(img):
    # Convert the image to a format suitable for OpenCV
    img_cv = np.array(img)
    
    # A global variable to store the selected bounding box coordinates
    bbox = []

    # Mouse callback function to handle user interaction
    def draw_rectangle(event, x, y, flags, param):
        nonlocal bbox, img_cv
        if event == cv2.EVENT_LBUTTONDOWN:
            # Set the starting point for the bounding box
            bbox = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            # Set the end point of the bounding box and draw it
            bbox.append((x, y))
            cv2.rectangle(img_cv, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.imshow("Select Bounding Box", img_cv)

    # Show the image and allow the user to draw the bounding box
    cv2.imshow("Select Bounding Box", img_cv)
    cv2.setMouseCallback("Select Bounding Box", draw_rectangle)
    
    # Wait for the user to draw the box or reset
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key to confirm selection
            break
        elif key == ord('r'):  # 'r' key to reset the box
            bbox = []
            img_cv = np.array(img)  # Reset the image
            cv2.imshow("Select Bounding Box", img_cv)
            print("Bounding box reset. Please redraw it.")
    
    cv2.destroyAllWindows()

    # Return the bounding box coordinates
    if len(bbox) == 2:
        return (min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1]), 
                max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1]))
    else:
        return None

def select_points_interactively(img):
    """
    Allows the user to interactively select multiple points on an image.
    Left mouse button: add a 'positive' point (green).
    Right mouse button: add a 'negative' point (red).
    Press 'r' to reset all points.
    Press Enter (key code 13) to confirm and finish.

    Returns:
        List of tuples: [(x1, y1, label1), (x2, y2, label2), ...]
            where label=1 for positive, label=0 for negative.
    """

    # Convert the PIL image or other type to a NumPy array suitable for OpenCV
    img_cv = np.array(img)
    # Make a copy for resetting
    original_img = img_cv.copy()

    # List to store points and labels: (x, y, label)
    # label = 1 for positive, 0 for negative
    labeled_points = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal labeled_points, img_cv

        # Left click => Positive point
        if event == cv2.EVENT_LBUTTONDOWN:
            labeled_points.append((x, y, 1))
            cv2.circle(img_cv, (x, y), 4, (0, 255, 0), -1)  # Green circle
            cv2.imshow(window_name, img_cv)

        # Right click => Negative point
        elif event == cv2.EVENT_RBUTTONDOWN:
            labeled_points.append((x, y, 0))
            cv2.circle(img_cv, (x, y), 4, (0, 0, 255), -1)  # Red circle
            cv2.imshow(window_name, img_cv)

    # Create a window to display the image
    window_name = "Select Points (L-click=Positive, R-click=Negative)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_cv)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # If 'Enter' (ASCII 13) is pressed, exit
        if key == 13:
            break

        # If 'r' is pressed, reset
        if key == ord('r'):
            labeled_points = []
            img_cv = original_img.copy()
            cv2.imshow(window_name, img_cv)
            print("Points reset. Please re-select.")

    # Close the window
    cv2.destroyWindow(window_name)

    return labeled_points


# Function to apply the mask and create a transparent PNG
def save_with_transparent_mask(img, mask, save_path):
    # Convert the original image to RGBA (with an alpha channel)
    img_rgba = img.convert("RGBA")
    
    # Create a new image with the same size and transparent background
    img_with_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Convert the mask to the same size as the image
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    
    # Set the image alpha channel based on the mask
    img_with_mask.paste(img_rgba, mask=mask_resized)
    
    # Save the image with the transparent mask
    img_with_mask.save(save_path, "PNG")


img_folder = 'data/images/'
img_files = os.listdir(img_folder)
img_files = [f for f in img_files if f.endswith('.jpg')]

box_or_point = 'point'

for img_file in img_files:
    img = Image.open(img_folder + img_file)
    
    if box_or_point == 'box':
        # manually set bbox here
        prompt = select_bbox_interactively(img)
    elif box_or_point == 'point':
        prompt = select_points_interactively(img)

    mask = get_gripper_mask(img, prompt, box_or_point)
    # Save the mask as a separate file
    # os.makedirs('data/masks', exist_ok=True)
    # mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    # mask_image.save(f'data/masks/{img_file}')

    # Save the image with the transparent mask to a new folder
    os.makedirs('data/masks', exist_ok=True)
    save_with_transparent_mask(img, mask, f'data/masks/{img_file}')