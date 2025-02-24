import base64
import json
import zlib
from PIL import Image
import io
import numpy as np
import os

json_path = "gripper_anno/ann_ext"
mask_save_path = "data/images/masks_all"
ori_img_path = "data/images_ext"
tgt_img_path = "data/images"
img_ext = ".jpg"

json_files = os.listdir(json_path)
json_files = [f for f in json_files if f.endswith(".json")]

ori_img_files = os.listdir(ori_img_path)
ori_img_files = [f for f in ori_img_files if f.endswith(img_ext)]

if not os.path.exists(mask_save_path):
    os.makedirs(mask_save_path)

for json_file in json_files:
    json_data = json.load(open(os.path.join(json_path, json_file)))
    #  Extract image dimensions
    image_width = json_data["size"]["width"]
    image_height = json_data["size"]["height"]

    # Extract base64 data
    bitmap_info = json_data["objects"][0]["bitmap"]
    bitmap_data = bitmap_info["data"]
    origin_x, origin_y = bitmap_info["origin"]

    # Decode base64 and decompress
    compressed_data = base64.b64decode(bitmap_data)
    image_data = zlib.decompress(compressed_data)

    # Load the bitmap
    breakpoint()
    bitmap_image = Image.open(io.BytesIO(image_data)).convert("L")

    # Create a white image with the full size
    full_image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

    # Paste the bitmap onto the blank image at the correct position
    mask = Image.new("L", (image_width, image_height), 0)
    mask.paste(bitmap_image, (origin_x, origin_y))

    # # Paste the bitmap onto the blank image at the correct position
    # full_image.paste(bitmap_image, (origin_x, origin_y), bitmap_image)
    # full_image = full_image.convert("L")

    # crop the original image with the mask
    ori_img = Image.open(f"{ori_img_path}/{json_file.replace(f'{img_ext}.json', img_ext)}").convert("RGBA")
    ori_img = np.array(ori_img)
    mask = np.array(mask)
    alpha_channel = np.where(mask > 0, 255, 0).astype(np.uint8)
    ori_img[..., :3] = ori_img[..., :3] * (alpha_channel[..., None] // 255)

    # Apply the mask to the original image
    result_array = np.zeros_like(ori_img)
    result_array[..., :3] = ori_img[..., :3]  # Copy RGB channels
    result_array[..., 3] = alpha_channel  # Set alpha channel

    # Convert back to image and save
    result_image = Image.fromarray(result_array)
    result_image.save(f"{mask_save_path}/{json_file.replace(f'{img_ext}.json', '.png')}", "PNG")

    os.system(f"cp {ori_img_path}/{json_file.replace(f'{img_ext}.json', img_ext)} {tgt_img_path}/{json_file.replace(f'{img_ext}.json', img_ext)}")

