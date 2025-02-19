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
from gripper_classifier import GripperClassifier

def get_args_parser():
    parser = argparse.ArgumentParser("End-effector tracking")
    parser.add_argument("--data_path", type=str, default='/media/gvl/ACDA-BDB0/datasets/bridge/bridge_dataset/1.0.0')
    parser.add_argument("--trainval", type=str, default='val')
    ## DINO related
    parser.add_argument("--DINO_model", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=798)
    parser.add_argument("--classifier_model", type=str, default="gripper_classifier.pth")
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

def init_classifier(model_path, device):
    classifier = GripperClassifier(in_dim=768).to(device)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier

def get_ee_masks():
    pass

def main(args):
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

    classifier = init_classifier(args.classifier_model, device)

    b_tfds = tfds.builder_from_directory(builder_dir=args.data_path)
    ds = b_tfds.as_dataset(split=args.trainval)
    for episode in tqdm(ds):
        images = [step['observation']['image_0'] for step in episode['steps']]
        images = [Image.fromarray(img.numpy()).convert("RGB") for img in images]
        images = [base_transform(img) for img in images]
        batched_images = torch.stack(images).to(device)
        with torch.no_grad():
            dinov2_feats = model.forward_intermediates(
                batched_images, 
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True
            )[-1].permute(0, 2, 3, 1)

            print(dinov2_feats.shape)
            breakpoint()
        
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)