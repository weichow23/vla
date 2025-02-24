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
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import timm
import types
import torch
from gripper_classifier import GripperClassifier, BNHead
from data_preprocessing.utils import load_all_data, save_video_with_overlay, concat_videos_grid
from data_preprocessing.masks2traj import compute_trajectory
from data_preprocessing.correspondences2poses import intrinsics_resize
import gc
import json
import time
import math
import cv2
from huggingface_hub import hf_hub_download

# TEST classifier with relu

def get_args_parser():
    parser = argparse.ArgumentParser("Parallel End-effector tracking")
    parser.add_argument("--dataset_name", type=str, default="bridge_orig")
    parser.add_argument("--trainval", type=str, default="val")
    parser.add_argument("--data_path", type=str, default='/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/bridge_preprocessed')
    parser.add_argument("--mul_cam", action="store_true", help="Calibrate all cameras we have in the dataset")
    parser.add_argument("--DINO_model", type=str, default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=798)
    parser.add_argument("--batch_size", type=int, default=128)
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
    # denormalizer = transforms.Normalize(
    #     mean=[-m/s for m, s in zip(normalizer.mean, normalizer.std)],
    #     std=[1/s for s in normalizer.std]
    # )
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

    return model, base_transform

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
        return None
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
    errors = np.linalg.norm(ee_2d_traj[inliers] - projected_points.squeeze(), axis=1)
    mean_error = np.mean(errors)

    if not success:
        return None
    # rmat, _ = cv2.Rodrigues(rvec)
    # P = np.hstack((rmat, tvec))
    return (rvec.astype(np.float32), tvec.astype(np.float32), ori_inliers), (len(valid_idx), len(inliers), mean_error)

def main(args):
    np.random.seed(111)
    # ds = b_tfds.as_dataset(split=f"{args.trainval}[:1]")
    args.feat_dim = 768 if "base" in args.DINO_model else 1024
    episodes_metadata = json.load(open(os.path.join(args.data_path, 'samples_all_val_filtered.json')))

    if True:
        if not os.path.exists(args.vis_path):
            os.makedirs(args.vis_path)

    Image.MAX_IMAGE_PIXELS = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, base_transform = init_DINO_model(
        model_identifier=args.DINO_model,
        stride=args.stride,
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        device=device,
    )

    classifier = init_classifier(args.classifier_type, args.classifier_model, device, feat_dim=args.feat_dim, output_shape=(args.img_size, args.img_size)).eval()

    cam_intrinsics = get_intrinsics((args.img_size, args.img_size))

    total_data_time = 0
    total_solver_time = 0
    total_model_time = 0
    total_time = 0

    episode_list = list(episodes_metadata.keys())
    episode_list = np.random.permutation(episode_list)[:64]

    estimated_episode_metadata = {}

    for idx, episode in tqdm(enumerate(episode_list)):
        start= time.time()
        epi_path = os.path.join(args.data_path, f"{episode}.h5")
        epi_data = load_all_data(epi_path, keys=['rgb', 'action'], BGR2RGB=False)
        ee_6d_poses = epi_data['action'][:, 0, :6].astype(np.float32)

        for cam in epi_data['rgb'].keys():
            ori_images = epi_data['rgb'][cam]
            images = [Image.fromarray(img).convert("RGB") for img in ori_images]
            images = [base_transform(img) for img in images]
            batched_images = torch.stack(images).to(device)
            data_time = time.time() - start
            total_data_time += data_time

            model_start = time.time()
            gripper_masks = get_gripper_masks(model, classifier, batched_images, args.batch_size, feat_dim=args.feat_dim, feat_size=args.img_size//args.stride)
            ee_traj, modified_masks = compute_trajectory(gripper_masks.cpu().numpy())
            model_time = time.time() - model_start
            total_model_time += model_time

            if np.isnan(ee_traj).all():
                print(f"Episode: {episode}/{cam} has no valid trajectory.")
                estimated_episode_metadata[episode] = "No valid trajectory"
                # continue
            else:
                solver_start = time.time()
                result, log = get_cam_pose(ee_traj, ee_6d_poses[..., :3], cam_intrinsics)
                solver_time = time.time() - solver_start
                total_solver_time += solver_time

            if result is None:
                print(f"Episode: {episode}/{cam} has no valid camera pose.")
                estimated_episode_metadata[episode] = "No valid camera pose"
                # continue
            else:
                estimated_episode_metadata[episode] = {
                    "r": result[0].tolist(),
                    "t": result[1].tolist(),
                    "num_waypoints": log[0], # number of waypoints, might be less than len(images) due to undetected grippers
                    "num_inliers": log[1],
                    "mean_proj_error": log[2]
                }

            # model_time = time.time() - model_start

            # total_model_time += model_time
            # total_time += time.time() - start
            # print(f"Episode: {episode}, Length: {len(images)}, Time: {time.time() - start}, Model time: {model_time}")

            if args.visualize:
                temp_path = f"{args.vis_path}/temp/"
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                save_path = os.path.join(temp_path, f"{episode.replace('/', '_')}_{cam}.mp4")
                # save_video_with_overlay(
                #     ori_images,
                #     gripper_masks,
                #     ee_traj.copy(),
                #     fps=5,
                #     save_path=save_path
                # )
                save_video_with_overlay(
                    ori_images,
                    modified_masks,
                    ee_traj.copy(),
                    ee_6d_traj=ee_6d_poses,
                    cam_params=(cam_intrinsics, result[0], result[1]),
                    inliers=result[2],
                    fps=5,
                    save_path=save_path,
                    idx=idx
                )

            del batched_images, gripper_masks, ee_traj, modified_masks
            torch.cuda.empty_cache()
            gc.collect()

            if not args.mul_cam:
                break

    if args.visualize:
        videos = os.listdir(temp_path)
        videos = [os.path.join(temp_path, v) for v in videos]
        concat_videos_grid(videos, f"{args.vis_path}/all_videos.mp4")
        os.system(f"rm -rf {temp_path}")

    with open(f"{args.meta_path}/estimated_poses.json", "w") as f:
        json.dump(estimated_episode_metadata, f, indent=4)

    print(f"Avg data time: {total_data_time / len(episode_list)}")
    print(f"Avg model time: {total_model_time / len(episode_list)}")
    print(f"Avg solver time: {total_solver_time / len(episode_list)}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)