"""
Given a video, this script extracts the 2D tracks of the end-effector of the robot.
Let's try different mode:
1. Run DINOv2 + classifier frame-by-frame
2. Use DINOv2 + classifier to initialize CoTracker at the middle of the video (where the gripper should always be visible), then treat the ee traj as the centre point of all tracking points
3. Use DINOv2 + classifier to initialize SAM2 at the middle of the video (where the gripper should always be visible), then treat the ee traj as the centre point of tracked mask in each frame
"""
import argparse
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import timm
import types
import torch
from gripper_classifier import GripperClassifier, BNHead
from data_preprocessing.utils import load_all_data, save_video_with_overlay
from data_preprocessing.masks2traj import compute_trajectory
import gc
import json
import os
import time
import math

def get_args_parser():
    parser = argparse.ArgumentParser("End-effector tracking")
    parser.add_argument("--data_path", type=str, default='/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/bridge_preprocessed/') # /bridge_dataset/1.0.0
    parser.add_argument("--trainval", type=str, default='val')
    ## DINO related
    parser.add_argument("--DINO_model", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--classifier_model", type=str, default="gripper_classifier_v2.pth")
    parser.add_argument("--vis_path", type=str, default="vis/")
    return parser

def init_DINO_model(model_identifier, stride, img_size, patch_size, device):
    model = timm.create_model(
        model_identifier,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
    )
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

# def init_classifier(model_path, device):
#     classifier = GripperClassifier(in_dim=768).to(device)
#     classifier.load_state_dict(torch.load(model_path))
#     classifier.eval()
#     return classifier

def init_classifier(model_path, device, output_shape=(256, 256)):
    classifier = BNHead(num_channels=768, output_shape=output_shape).to(device)
    # classifier.load_state_dict(torch.load(model_path))
    return classifier

@torch.inference_mode()
def get_gripper_masks(model, classifier, batched_images, batch_size, threshold=0.5):
    classifier_outputs = []
    chunck_num = len(batched_images) // batch_size
    chunck_num = chunck_num + 1 if len(batched_images) % batch_size != 0 else chunck_num
    chunck_size = math.ceil(len(batched_images) / chunck_num)
    for i in range(0, len(batched_images), chunck_size):
        one_batch = batched_images[i:i+chunck_size] # bs, 3, 798, 798
        with torch.no_grad():
            # dinov2_feats = model.forward_intermediates(
            #     one_batch, 
            #     norm=True,
            #     output_fmt="NCHW",
            #     intermediates_only=True
            # )[-1].permute(0, 2, 3, 1)
            dinov2_feats = model.forward_features(one_batch)[:, 1:]
            dinov2_feats = dinov2_feats.permute(0, 2, 1).reshape(-1, 768, 37, 37)

            classifier_output = classifier(dinov2_feats).squeeze()
            classifier_outputs.append(classifier_output)
            del dinov2_feats
            torch.cuda.empty_cache()

    classifier_outputs = torch.cat(classifier_outputs, dim=0)
    classifier_outputs = classifier_outputs > threshold

    return classifier_outputs

def main(args):
    # b_tfds = tfds.builder('bridge_dataset', data_dir=args.data_path)
    # ds = b_tfds.as_dataset(split=f"{args.trainval}[:1]")

    episodes_metadata = json.load(open(os.path.join(args.data_path, 'samples_all_val_filtered.json')))

    if True:
        if not os.path.exists(args.vis_path):
            os.makedirs(args.vis_path)

    Image.MAX_IMAGE_PIXELS = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, base_transform, denormalizer = init_DINO_model(
        model_identifier=args.DINO_model,
        stride=args.stride,
        img_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
        device=device,
    )

    classifier = init_classifier(args.classifier_model, device, output_shape=(args.img_size, args.img_size)).eval()

    total_time = 0
    total_model_time = 0

    episode_list = list(episodes_metadata.keys())

    for idx, episode in tqdm(enumerate(episode_list)):
        start= time.time()
        epi_path = os.path.join(args.data_path, f"{episode}.h5")
        ori_images = load_all_data(epi_path, keys=['rgb'], cameras=['image_0'], BGR2RGB=False)['rgb']['image_0']
        images = [Image.fromarray(img).convert("RGB") for img in ori_images]
        images = [base_transform(img) for img in images]
        batched_images = torch.stack(images).to(device)

        model_start = time.time()

        gripper_masks = get_gripper_masks(model, classifier, batched_images, args.batch_size)
        ee_traj, modified_masks = compute_trajectory(gripper_masks.cpu().numpy())

        model_time = time.time() - model_start

        total_model_time += model_time
        total_time += time.time() - start
        print(f"Episode: {episode}, Length: {len(images)}, Time: {time.time() - start}, Model time: {model_time}")

        if True: # vis
            save_path = os.path.join(args.vis_path, f"{episode.replace('/', '_')}.mp4")
            save_video_with_overlay(
                ori_images,
                gripper_masks,
                ee_traj.copy(),
                fps=5,
                save_path=save_path
            )
            save_video_with_overlay(
                ori_images,
                modified_masks,
                ee_traj.copy(),
                fps=5,
                save_path=save_path.replace('.mp4', '_filtered.mp4')
            )

        del batched_images, gripper_masks, ee_traj, modified_masks
        torch.cuda.empty_cache()
        gc.collect()

        if idx == 10:
            break

    print(f"Avg time: {total_time/20}, Avg model time: {total_model_time/20}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)