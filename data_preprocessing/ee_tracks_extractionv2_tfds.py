"""
Given a video, this script extracts the 2D tracks of the end-effector of the robot.
Let's try different mode:
1. Run DINOv2 + classifier frame-by-frame
2. Use DINOv2 + classifier to initialize CoTracker at the middle of the video (where the gripper should always be visible), then treat the ee traj as the centre point of all tracking points
3. Use DINOv2 + classifier to initialize SAM2 at the middle of the video (where the gripper should always be visible), then treat the ee traj as the centre point of tracked mask in each frame
"""
import os
import sys
# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import timm
import types
import torch
from gripper_classifier import GripperClassifier, BNHead
from data_preprocessing.utils import load_all_data, save_video_with_overlay, concat_videos_grid, add_data_hdf5
from data_preprocessing.masks2traj import compute_trajectory
from data_preprocessing.correspondences2poses import intrinsics_resize
import gc
import json
import time
import math
import cv2
from huggingface_hub import hf_hub_download
import dlimp as dl
from torch.utils.data import Dataset, DataLoader, get_worker_info, IterableDataset
from dataclasses import dataclass
import jsonlines
import h5py
from datetime import datetime

   
class RLDSDataset(IterableDataset):
    def __init__(
            self,
            data_path,
            base_transform,
            dataset_name="bridge_orig",
            trainval="train",
            progress=None
    ):
        self.data_path = data_path
        self.trainval = trainval
        with tf.device('/CPU:0'):
            builder = tfds.builder(dataset_name, data_dir=self.data_path)
            dataset = dl.DLataset.from_rlds(builder, split=trainval, shuffle=False, num_parallel_reads=tf.data.AUTOTUNE)
            dataset = dataset.ignore_errors()
        self.dataset = dataset.with_ram_budget(1)
        self.base_transform = base_transform
        self.progress = progress

    def process_episode(self, episode):
        """Processes a single episode and applies transformations."""
        output = {}
        image_names = {key[6:] for key in episode["observation"] if key.startswith("image_")}
        output["episode_id"] = torch.tensor(episode["traj_metadata"]["episode_metadata"]["episode_id"][0])
        output["states"] = torch.tensor(episode['observation']['state'])
        output["cameras"] = {}
        output["episode_file"] = episode['traj_metadata']['episode_metadata']['file_path'][0].decode('utf-8')

        for name in image_names:
            images = episode["observation"][f"image_{name}"]
            images = [tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8) for image in images]
            images = [Image.fromarray(image.numpy()).convert("RGB") for image in images]
            images = [self.base_transform(img) for img in images]
            batched_images = torch.stack(images)
            output["cameras"][name] = batched_images

        return output

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Handles multiprocessing by partitioning data among workers."""
        with tf.device('/CPU:0'):
            worker_info = get_worker_info()
            dataset_iterator = self.dataset.as_numpy_iterator()
        
            if worker_info is None:
                # Single-worker mode, return the full dataset
                for episode in dataset_iterator:
                    if self.progress is not None:
                        episode_file = episode['traj_metadata']['episode_metadata']['file_path'][0].decode('utf-8')
                        episode_id = episode["traj_metadata"]["episode_metadata"]["episode_id"][0]
                        unique_id = f"{episode_file}~{episode_id}"
                        if unique_id in self.progress:
                            # print(f"skip {unique_id}")
                            yield {}
                        else:
                            yield self.process_episode(episode)
                    else:
                        yield self.process_episode(episode)
            else:
                # Multi-worker mode: partition dataset among workers
                worker_id, num_workers = worker_info.id, worker_info.num_workers
                for i, episode in enumerate(dataset_iterator):
                    if self.progress is not None:
                        episode_file = episode['traj_metadata']['episode_metadata']['file_path'][0].decode('utf-8')
                        episode_id = episode["traj_metadata"]["episode_metadata"]["episode_id"][0]
                        unique_id = f"{episode_file}~{episode_id}"
                        if i % num_workers == worker_id:  # Assign episodes round-robin
                            if unique_id in self.progress:
                                # print(f"skip {unique_id}")
                                yield {}
                            else:
                                yield self.process_episode(episode)
                    else:
                        if i % num_workers == worker_id:
                            yield self.process_episode(episode)

def get_args_parser():
    parser = argparse.ArgumentParser("Parallel End-effector tracking")
    parser.add_argument("--dataset_name", type=str, default="bridge_orig")
    parser.add_argument("--trainval", type=str, default="val")
    parser.add_argument("--num_episodes", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default='/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/OXE') # bridge_preprocessed
    parser.add_argument("--mul_cam", action="store_true", help="Calibrate all cameras we have in the dataset")
    parser.add_argument("--DINO_model", type=str, default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=798)
    parser.add_argument("--batch_size", type=int, default=128, help="this is actually the number of maximun images to process by DINOv2 at once")
    parser.add_argument("--classifier_type", type=str, default="BNHead")
    parser.add_argument("--classifier_model", type=str, default="data_preprocessing/classifier_ckpt/gripper_classifier_large_conv_v2_aug.pth")
    parser.add_argument("--meta_path", type=str, default="data_preprocessing/meta_data/")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_path", type=str, default="vis/")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    return parser

def init_DINO_model(model_identifier, stride, img_size, patch_size, device):
    model = timm.create_model(
        model_identifier,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        )
    
    # # Uncomment to load DVT
    # # Define repo and model file path
    # repo_id = "jjiaweiyang/DVT"  # Your Hugging Face repo
    # filename = "imgnet_distilled/vit_base_patch14_dinov2.lvd142m.pth"  # Path within repo
    # # Download the checkpoint
    # checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # # Extract the actual state_dict (handling cases where it's nested un der 'model' key)
    # if "model" in checkpoint:
    #     checkpoint = checkpoint["model"]  # Extract model state_dict
    # checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
    # model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    # Different models have different data configurations
    # e.g., their training resolution, normalization, etc, are different
    data_config = timm.data.resolve_model_data_config(model=model)
    transformation = timm.data.create_transform(**data_config, is_training=False)
    normalizer = transformation.transforms[-1]
    denormalizer = transforms.Normalize(
        mean=[-m/s for m, s in zip(normalizer.mean, normalizer.std)],
        std=[1/s for s in normalizer.std]
    )
    base_transform = transforms.Compose(
        [
            transforms.Resize(img_size, Image.BICUBIC, antialias=True),
            transforms.ToTensor(),
            normalizer,
        ]
    )

    if stride != model.patch_embed.proj.stride[0]:
        model.patch_embed.proj.stride = [stride, stride]

        def dynamic_feat_size(self, img_size):
            """Get grid (feature) size for given image size taking account of dynamic padding.
            NOTE: must be torchscript compatible so using fixed tuple indexing
            """
            return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                img_size[1] - self.patch_size[1]
            ) // self.proj.stride[1] + 1

        model.patch_embed.dynamic_feat_size = types.MethodType(
            dynamic_feat_size, model.patch_embed
        )

    return model, base_transform, denormalizer

def init_classifier(model_type, model_path, device, feat_dim=768, output_shape=(256, 256)):
    if model_type == "GripperClassifier":
        classifier = GripperClassifier(in_dim=feat_dim, output_shape=output_shape).to(device)
    else:
        classifier = BNHead(num_channels=feat_dim, output_shape=output_shape).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    return classifier.eval()

@torch.inference_mode()
def get_gripper_masks(model, classifier, batched_images, batch_size, feat_dim=768, feat_size=37, threshold=0.55):
    classifier_outputs = []
    chunck_num = len(batched_images) // batch_size
    chunck_num = chunck_num + 1 if len(batched_images) % batch_size != 0 else chunck_num
    chunck_size = math.ceil(len(batched_images) / chunck_num)
    for i in range(0, len(batched_images), chunck_size):
        one_batch = batched_images[i:i+chunck_size] # bs, 3, 798, 798
        with torch.no_grad():
            dinov2_feats = model.forward_features(one_batch)[:, 1:]
            dinov2_feats = dinov2_feats.permute(0, 2, 1).reshape(-1, feat_dim, feat_size, feat_size)

            classifier_output = classifier(dinov2_feats).squeeze()
            classifier_outputs.append(classifier_output)
            del dinov2_feats
            torch.cuda.empty_cache()

    classifier_outputs = torch.cat(classifier_outputs, dim=0) > threshold
    return classifier_outputs

def get_intrinsics(img_size):
    init_intrinsic=np.array(
                    [[623.588, 0, 319.501], 
                     [0, 623.588, 239.545], 
                     [0, 0, 1]], dtype=np.float32
                ) # logitech C920
    init_intrinsic = intrinsics_resize(init_intrinsic, (640, 480), img_size)
    return init_intrinsic

def get_cam_pose(ee_2d_traj, ee_3d_traj, cam_intrinsics):
    assert len(ee_2d_traj) == len(ee_3d_traj)
    valid_idx = np.where(~np.isnan(ee_2d_traj[:, 0]))[0]
    ee_2d_traj = ee_2d_traj[valid_idx]
    ee_3d_traj = ee_3d_traj[valid_idx]

    if len(ee_2d_traj) <= 4:
        return None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        ee_3d_traj,       # 3D points in the object’s coordinate space
        ee_2d_traj,        # Corresponding 2D projections in the image
        cam_intrinsics,       # Camera intrinsic matrix
        None,         # Distortion coefficients
        rvec=np.array([2.127825975418091, -1.2211724519729614, 0.36597317457199097]),
        tvec=np.array([-0.16111209988594055, 0.22177191078662872, 0.3002764582633972]),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,      # or cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_EPNP, etc.
        iterationsCount=1000,          # RANSAC iterations
        reprojectionError=40.0,        # RANSAC reprojection threshold (pixels) NOTE may need to adjust this
        confidence=0.999               # RANSAC confidence
    )
    ori_inliers = valid_idx[inliers]

    rvec, tvec = cv2.solvePnPRefineLM(
        ee_3d_traj[inliers],
        ee_2d_traj[inliers],
        cam_intrinsics,
        None,
        rvec, 
        tvec
    )

    ## let's also log the reprojection error 
    projected_points, _ = cv2.projectPoints(
        ee_3d_traj[inliers],  # Inlier 3D points
        rvec, 
        tvec, 
        cam_intrinsics, 
        None
    )

    # Compute the reprojection error
    errors = np.linalg.norm(ee_2d_traj[inliers].squeeze() - projected_points.squeeze(), axis=1)
    mean_error = np.mean(errors)

    if not success:
        return None, None
    # rmat, _ = cv2.Rodrigues(rvec)
    # P = np.hstack((rmat, tvec))
    return (rvec.astype(np.float32), tvec.astype(np.float32), ori_inliers), (len(valid_idx), len(inliers), mean_error)

def main(args):
    torch.manual_seed(111)
    np.random.seed(111)
    args.feat_dim = 768 if "base" in args.DINO_model else 1024

    if True:
        if not os.path.exists(args.vis_path):
            os.makedirs(args.vis_path)

    Image.MAX_IMAGE_PIXELS = None
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, base_transform, denormalizer = init_DINO_model(
        model_identifier=args.DINO_model,
        stride=args.stride,
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        device=device,
    )

    classifier = init_classifier(args.classifier_type, args.classifier_model, device, feat_dim=args.feat_dim, output_shape=(args.img_size, args.img_size)).eval()

    cam_intrinsics = get_intrinsics((args.img_size, args.img_size))

    args.meta_path = f"{args.meta_path}/{args.trainval}"
    if not os.path.exists(args.meta_path):
        os.makedirs(args.meta_path)

    # debug only
    if args.num_episodes > 0:
        total_size = args.num_episodes
        chunk_size = total_size // args.num_gpus
        start_idx = args.gpu_id * chunk_size #  is the number of episodes already processed
        end_idx = start_idx + chunk_size
        # if os.path.exists(f"{args.meta_path}/estimated_poses{args.gpu_id}.jsonl"):
        #     with open(f"{args.meta_path}/estimated_poses{args.gpu_id}.jsonl", "r") as f:
        #         line_count = sum(1 for _ in f)
        #     if line_count >= chunk_size:
        #         print(f"GPU {args.gpu_id} already processed {line_count} episodes")
        #         return
        #     start_idx = start_idx + line_count
        args.trainval = f"{args.trainval}[{start_idx}:{end_idx}]"

    progress = None
    if os.path.exists(f"{args.meta_path}/progress_dict.json"):
        with open(f"{args.meta_path}/progress_dict.json", "r") as f:
            progress = json.load(f)
        progress = progress.keys()

    dataset = RLDSDataset(
        data_path=args.data_path,
        base_transform=base_transform,
        dataset_name=args.dataset_name,
        trainval=args.trainval,
        progress=progress
    )
    # print(f"Running {len(dataset)} episodes on GPU {args.gpu_id}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        pin_memory=True,
        # persistent_workers=True,
    )

    estimated_episode_metadata = {}

    save_freq = 200
    if not os.path.exists(f"{args.data_path}/raw_cam_vla"):
        os.makedirs(f"{args.data_path}/raw_cam_vla")

    progress_files = os.listdir(f"{args.meta_path}")
    progress_files = [f for f in progress_files if f.endswith(".jsonl")]

    for f in progress_files:
        with jsonlines.open(f"{args.meta_path}/{f}") as f:
            for line in f:
                estimated_episode_metadata.update(line)
                
    print(f"Processed {len(estimated_episode_metadata.keys())} episodes")

    for iter, batch in enumerate(tqdm(dataloader, desc=f"GPU {args.gpu_id}", position=args.gpu_id, leave=True)): # , total=len(dataset)
        if len(batch) == 0:
            continue
        for idx in range(batch['episode_id'].shape[0]):
            episode_id = int(batch['episode_id'][idx])
            episode_file = batch['episode_file'][idx]
            unique_id = f"{episode_file}~{episode_id}"

            estimated_episode_metadata.setdefault(unique_id, {})
            
            ee_6d_poses = batch['states'][idx].numpy()

            for cam in batch['cameras'].keys():
                if not args.mul_cam and cam != "0":
                    continue
                if cam in estimated_episode_metadata[unique_id]:
                    print(f"GPU {args.gpu_id}: Episode {unique_id} already processed for camera {cam}")
                    continue
                batched_images = batch['cameras'][cam][idx].to(device)

                gripper_masks = get_gripper_masks(model, classifier, batched_images, args.batch_size, feat_dim=args.feat_dim, feat_size=args.img_size//args.stride)
                ee_traj, modified_masks = compute_trajectory(gripper_masks.cpu().numpy())

                result = None
                if not np.isnan(ee_traj).all():
                    result, log = get_cam_pose(ee_traj, ee_6d_poses[..., :3], cam_intrinsics)
                else:
                    ee_traj = None

                if result is None:
                    estimated_episode_metadata[unique_id][cam] = None
                else:
                    estimated_episode_metadata[unique_id][cam] = {
                        "r": result[0].tolist(),
                        "t": result[1].tolist(),
                        "num_waypoints": log[0], # number of waypoints, might be less than len(images) due to undetected grippers
                        "num_inliers": log[1],
                        "mean_proj_error": log[2]
                    }

                if args.visualize and iter % save_freq == 0: # Save video every 200 episodes
                    temp_path = os.path.join(args.vis_path, "temp")
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                    save_path = os.path.join(temp_path, f"{unique_id.replace('/', '_')}_{cam}.mp4")
                    ori_images = [(denormalizer(img).cpu().numpy()*255).astype(np.uint8).transpose(1,2,0) \
                                  for img in batched_images]
                    save_video_with_overlay(
                        ori_images,
                        modified_masks,
                        ee_traj.copy(),
                        ee_6d_traj=ee_6d_poses,
                        cam_params=(cam_intrinsics, result[0], result[1]) if result is not None else None,
                        inliers=result[2] if result is not None else None,
                        fps=5,
                        save_path=save_path,
                        idx=idx
                    )

                raw_estimated_data = {}
                raw_estimated_data[unique_id] = {}
                raw_estimated_data[unique_id][cam] = estimated_episode_metadata[unique_id][cam].copy() \
                                        if estimated_episode_metadata[unique_id][cam] is not None else {}
                raw_estimated_data[unique_id][cam].update(
                    {
                        "ee_traj": ee_traj,
                        "inliers": result[2] if result is not None else None,
                    }
                )
                add_data_hdf5(raw_estimated_data, f"{args.data_path}/raw_cam_vla/data_gpu_{args.gpu_id}.h5")

                del batched_images, gripper_masks, ee_traj, modified_masks
                torch.cuda.empty_cache()
                gc.collect()

            with jsonlines.open(f"{args.meta_path}/estimated_poses{args.gpu_id}.jsonl", "a") as f:
                f.write({
                    unique_id: estimated_episode_metadata[unique_id]
                })

            # if len(raw_estimated_data) % save_freq == 0:
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     filename = f"{args.data_path}/raw_cam_vla/data_{timestamp}_gpu_{args.gpu_id}.h5"       
            #     with h5py.File(filename, "w") as f:
            #         save_dict_to_hdf5(raw_estimated_data, f)
            #     raw_estimated_data = {}

    # if args.visualize:
    #     videos = os.listdir(temp_path)
    #     videos = [os.path.join(temp_path, v) for v in videos]
    #     concat_videos_grid(videos, f"{args.vis_path}/all_videos.mp4")
    #     os.system(f"rm -rf {temp_path}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)