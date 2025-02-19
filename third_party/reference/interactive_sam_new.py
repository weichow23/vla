import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def get_gripper_mask(img, prompt, box_or_point='box'):
    if box_or_point == 'box':
        # 'prompt' is a bounding box: (x1, y1, x2, y2)
        inputs = sam_processor(
            img, 
            input_boxes=[[prompt]],  # must be 2D for batch + multiple boxes
            return_tensors="pt"
        )
    elif box_or_point == 'point':
        # 'prompt' is a list of (x, y, label)
        # We need separate arrays for points and labels
        points = [[(x, y) for x, y, _lbl in prompt]]
        labels = [[_lbl for _x, _y, _lbl in prompt]]
        inputs = sam_processor(
            img,
            input_points=points,
            input_labels=labels,
            return_tensors="pt"
        )

    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Post-processing to get the mask in (H, W) shape
    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, 
        inputs["original_sizes"], 
        inputs["reshaped_input_sizes"]
    )[0][0][0].numpy()

    return mask

def overlay_mask_on_image(img_cv, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Overlays the mask on the original BGR image (img_cv) with a given alpha.
    'color' is in BGR format by default for OpenCV.
    """
    # Ensure mask is boolean or 0/1
    mask_bool = (mask > 0.5).astype(np.uint8)

    # Create a 3-channel colored mask
    color_mask = np.zeros_like(img_cv, dtype=np.uint8)
    color_mask[:, :] = color
    # Apply mask
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_bool)

    # Blend between original and color_mask
    blended = cv2.addWeighted(color_mask, alpha, img_cv, 1 - alpha, 0)
    return blended

def select_points_interactively_with_live_inference(img, sam_model, sam_processor):
    """
    Interactively select multiple points on an image and see the SAM mask update
    in real-time. 
      - Left mouse button: add a 'positive' point (label=1) (green circle)
      - Right mouse button: add a 'negative' point (label=0) (red circle)
      - Press 'z' to remove the last added point
      - Press 'r' to reset all points
      - Press Enter (key code 13) to confirm and finish

    Returns:
        List of (x, y, label), where label=1 for positive, 0 for negative.
    """

    # Convert the PIL image to a NumPy BGR array suitable for OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    original_img = img_cv.copy()

    # List to store labeled points: (x, y, label)
    labeled_points = []

    # A small function to update the mask given the current points
    def update_display():
        nonlocal labeled_points, img_cv
        # Start from a fresh copy
        img_cv = original_img.copy()

        # If we have points, run inference and overlay the mask
        if len(labeled_points) > 0:
            mask = get_gripper_mask(img, labeled_points, box_or_point='point')
            img_cv_masked = overlay_mask_on_image(img_cv, mask, alpha=0.5, color=(0, 255, 0))
            img_cv = img_cv_masked

        # Draw circles for each selected point
        for (px, py, lbl) in labeled_points:
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(img_cv, (px, py), 4, color, -1)

        cv2.imshow(window_name, img_cv)

    # Mouse callback
    def mouse_callback(event, x, y, flags, param):
        nonlocal labeled_points
        # Left click => Positive point
        if event == cv2.EVENT_LBUTTONDOWN:
            labeled_points.append((x, y, 1))
            update_display()

        # Right click => Negative point
        elif event == cv2.EVENT_RBUTTONDOWN:
            labeled_points.append((x, y, 0))
            update_display()

    # Window name
    window_name = "SAM Point Prompt: L-click=Positive, R-click=Negative, z=Undo, r=Reset, Enter=Finish"
    # Create a named window and resize to the desired size for easier viewing
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2000, 2000)

    cv2.setMouseCallback(window_name, mouse_callback)

    # Initial display
    cv2.imshow(window_name, img_cv)

    while True:
        key = cv2.waitKey(1) & 0xFF

        # If Enter (13) is pressed, exit
        if key == 13:
            break

        # If 'r' is pressed, reset
        elif key == ord('r'):
            labeled_points = []
            update_display()
            print("Points reset. Please re-select.")

        # If 'z' is pressed, remove the last point
        elif key == ord('z'):
            if len(labeled_points) > 0:
                labeled_points.pop()
                update_display()
                print("Last point removed.")

    cv2.destroyWindow(window_name)
    return labeled_points

def save_with_transparent_mask(img, mask, save_path):
    """
    Save the PIL image 'img' with a given mask as RGBA,
    so that the masked area is opaque and outside the mask is fully transparent.
    """
    img_rgba = img.convert("RGBA")
    
    # Create a new image with the same size and transparent background
    img_with_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Convert the mask to the same size as the image
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        img.size, Image.BILINEAR
    )
    
    # Paste using the mask as alpha
    img_with_mask.paste(img_rgba, mask=mask_resized)
    
    # Save
    img_with_mask.save(save_path, "PNG")


# -----------
# Example usage:
if __name__ == "__main__":
    import os

    img_folder = 'data/images/'
    img_files = os.listdir(img_folder)
    img_files = [f for f in img_files if f.endswith('.jpg')]

    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path)

        # Interactively select points and see real-time mask updates
        prompt_points = select_points_interactively_with_live_inference(
            img, sam_model, sam_processor
        )
        print("Final point prompts:", prompt_points)

        # Once the user finishes, get the final mask
        final_mask = get_gripper_mask(img, prompt_points, box_or_point='point')

        # Save the result as transparent PNG
        os.makedirs('data/masks', exist_ok=True)
        out_path = os.path.join('data/masks', img_file.replace('.jpg', '_mask.png'))
        save_with_transparent_mask(img, final_mask, out_path)
        print("Saved:", out_path)
