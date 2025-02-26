import argparse
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import timm
import types
import torch
from gripper_classifier import GripperClassifier
from data_preprocessing.utils import load_all_data, save_video_with_overlay
from data_preprocessing.masks2traj import compute_trajectory
import gc
import json
import os
import time
import math

def get_args_parser():
    parser = argparse.ArgumentParser("End-effector tracking")
    parser.add_argument("--data_path", type=str, default='STORRM/bridge_preprocessed/') # /bridge_dataset/1.0.0
    parser.add_argument("--trainval", type=str, default='val')
    ## DINO related
    parser.add_argument("--DINO_model", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=798)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--classifier_model", type=str, default="gripper_classifier_new.pth")
    parser.add_argument("--vis_path", type=str, default="vis/")
    return parser

def main(args):
    # b_tfds = tfds.builder('bridge_dataset', data_dir=args.data_path)
    # ds = b_tfds.as_dataset(split=f"{args.trainval}[:1]")

    episodes_metadata = json.load(open(os.path.join(args.data_path, 'samples_all_val_filtered.json')))

    np.random.seed(110)

    episodes = list(episodes_metadata.keys())
    episodes = np.random.choice(episodes, 200, replace=False)

    for idx, episode in tqdm(enumerate(episodes)):
        epi_path = os.path.join(args.data_path, f"{episode}.h5")
        ori_images = load_all_data(epi_path, keys=['rgb'], cameras=['image_0'], BGR2RGB=False)['rgb']['image_0']
        images = [Image.fromarray(img).convert("RGB") for img in ori_images]
        
        random_img = images[np.random.randint(0, len(images))]
        random_img.save(os.path.join("data/test", f"{episode.replace('/', '_')}.jpg"))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)