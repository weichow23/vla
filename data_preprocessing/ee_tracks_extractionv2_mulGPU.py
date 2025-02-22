import os
import sys
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import timm
import json
import gc
import math
from data_preprocessing.utils import load_all_data, save_video_with_overlay, concat_videos_grid
from data_preprocessing.masks2traj import compute_trajectory
from data_preprocessing.ee_tracks_extractionv2 import init_DINO_model, init_classifier, get_gripper_masks, get_args_parser
from gripper_classifier import GripperClassifier, BNHead

def process_episodes(rank, args, episode_list, vis_path=None):
    """ Each process handles a subset of the episodes on a specific GPU """
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # Load model and classifier
    model, base_transform = init_DINO_model(
        args.DINO_model, args.stride, (args.img_size, args.img_size), args.patch_size, device
    )
    classifier = init_classifier(args.classifier_type, 
                                 args.classifier_model, 
                                 device, 
                                 feat_dim=args.feat_dim,
                                 output_shape=(args.img_size, args.img_size))

    for idx, episode in tqdm(enumerate(episode_list), position=rank):
        epi_path = os.path.join(args.data_path, f"{episode}.h5")
        ori_images = load_all_data(epi_path, keys=['rgb'], cameras=['image_0'], BGR2RGB=False)['rgb']['image_0']
        images = [base_transform(Image.fromarray(img).convert("RGB")) for img in ori_images]
        batched_images = torch.stack(images).to(device)
        
        gripper_masks = get_gripper_masks(model, classifier, batched_images, args.batch_size, feat_dim=args.feat_dim, feat_size=args.img_size//args.stride)
        ee_traj, modified_masks = compute_trajectory(gripper_masks.cpu().numpy())

        if np.isnan(ee_traj).all():
            print(f"Skipping episode {episode}: No valid trajectory.")
            continue

        if args.visualize and idx % 1 == 0:
            # Save visualization every episode
            os.makedirs(vis_path, exist_ok=True)
            save_path = os.path.join(vis_path, f"{episode.replace('/', '_')}.mp4")
            save_video_with_overlay(
                ori_images, modified_masks, ee_traj.copy(), fps=5, save_path=save_path.replace('.mp4', '_filtered.mp4'), idx=idx
            )

        del batched_images, gripper_masks, ee_traj, modified_masks
        torch.cuda.empty_cache()
        gc.collect()

def main(args):
    np.random.seed(110)
    # Load episode metadata
    episodes_metadata = json.load(open(os.path.join(args.data_path, 'samples_all_val_filtered.json')))
    episode_list = list(episodes_metadata.keys())
    episode_list = np.random.permutation(episode_list)[:64]  # Shuffle episodes
    
    num_gpus = args.num_gpus
    processes = []

    args.feat_dim = 768 if "base" in args.DINO_model else 1024

    # Split work among GPUs
    episode_splits = np.array_split(episode_list, num_gpus)

    temp_path = None # Temporary directory for storing videos
    if args.visualize:
        temp_path = f"{args.vis_path}/temp"
        os.makedirs(temp_path, exist_ok=True)

    # Start multiple processes
    for rank in range(num_gpus):
        p = mp.Process(target=process_episodes, args=(rank, args, episode_splits[rank], temp_path))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    if args.visualize:
        # Merge results into a single video
        videos = [os.path.join(temp_path, v) for v in os.listdir(temp_path) if 'filtered' in v]
        final_output = f"{args.vis_path}/all_videos.mp4"
        concat_videos_grid(videos, final_output)

        # Cleanup temporary directories
        for rank in range(num_gpus):
            os.system(f"rm -rf {temp_path}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)  # Necessary for multiprocessing in some environments
    main(args)
