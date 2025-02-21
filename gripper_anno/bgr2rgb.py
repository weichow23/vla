import os
from PIL import Image
import numpy as np

bgr_files = os.listdir("data/images_ext")
bgr_files = [f for f in bgr_files if f.endswith(".jpg")]

for bgr_file in bgr_files:
    img = Image.open(f"data/images_ext/{bgr_file}")
    img = img.convert("RGB")
    img = np.array(img)[..., ::-1]
    img = Image.fromarray(img)
    img.save(f"data/img_ext/{bgr_file}")