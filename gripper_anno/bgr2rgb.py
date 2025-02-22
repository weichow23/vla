import os
from PIL import Image
import numpy as np

bgr_files = os.listdir("data/bridge_demo_set")
bgr_files = [f for f in bgr_files if f.endswith(".png")]

for bgr_file in bgr_files:
    img = Image.open(f"data/bridge_demo_set/{bgr_file}")
    img = img.convert("RGB")
    img = np.array(img)[..., ::-1]
    img = Image.fromarray(img)
    img.save(f"data/bridge_demo_set/{bgr_file}")