"""
Precompute the DINO v2 features and labels for training the gripper classifier.
"""
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
from data_preprocessing.ee_tracks_extraction import init_DINO_model
import torchvision.transforms.functional as TF

# Configuration
Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_identifier = "vit_base_patch14_dinov2.lvd142m"
stride = 4
img_size = 798
patch_size = 14
batch_size = 16  # Batch processing
feat_size = 197
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(110)
torch.manual_seed(110)

# Initialize DINO model
model, base_transform, denormalizer = init_DINO_model(
    model_identifier=model_identifier,
    stride=stride,
    img_size=(img_size, img_size),
    patch_size=patch_size,
    device=device,
)

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        img_files = os.listdir(img_dir)
        self.img_files = [f for f in img_files if f.endswith(".jpg") or f.endswith(".png")]
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace(".jpg", ".png"))

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert RGBA mask to single-channel mask by keeping only the alpha channel
        mask = np.array(mask)
        mask = mask * mask[..., -1:]
        mask = Image.fromarray(mask).convert("L")

        # --- Optional data augmentation ---
        if self.augment:
            angle = np.random.uniform(-100, 100)
            translate = (
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10)
            )
            scale = np.random.uniform(0.9, 1.1)
            shear = np.random.uniform(-10, 10)

            # Apply the same affine transformation to both image and mask
            img = TF.affine(
                img, angle=angle, translate=translate, scale=scale, shear=shear,
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.affine(
                mask, angle=angle, translate=translate, scale=scale, shear=shear,
                interpolation=InterpolationMode.NEAREST
            )

            # Ensure the mask is still valid (i.e., object not lost)
            mask_np_aug = np.array(mask)
            if mask_np_aug.max() == 0:  # Object disappeared
                return self.__getitem__(idx)  # retry with a new transformation

            # 50% chance of horizontal flip
            if np.random.rand() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        # --- End of optional data augmentation ---

        # Define the image transform: Resize → ToTensor → Normalize
        image_transform = transforms.Compose([
            transforms.Resize((798, 798), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Define the mask transform: Resize (Nearest) → ToTensor
        # (no normalization for mask)
        mask_transform = transforms.Compose([
            transforms.Resize((197, 197), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # Apply transforms
        img = image_transform(img)
        mask = mask_transform(mask)
        mask = (mask > 0.).float()  # binarize mask

        return img, mask, img_file

# Load dataset
img_dir = "data/images/"
mask_dir = "data/images/masks_all/"
dataset = ImageDataset(img_dir, mask_dir, augment=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
prefix = "002"

# Feature extraction and saving
feature_save_path = "data/features/"
os.makedirs(feature_save_path, exist_ok=True)

model.eval()
with torch.no_grad():
    for imgs, masks, img_files in tqdm(dataloader, desc="Processing Batches"):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Forward pass through DINO model
        dinov2_feats = model.forward_intermediates(
            imgs, norm=True, output_fmt="NCHW", intermediates_only=True
        )[-1].permute(0, 2, 3, 1)  # Convert to [B, H, W, C]
        
        for i in range(len(img_files)):
            save_name = img_files[i].replace(".png", ".jpg") 
            feat_filename = os.path.join(feature_save_path, save_name.replace(".jpg", f"_{prefix}.pth"))
            mask_filename = os.path.join(feature_save_path, save_name.replace(".jpg", f"_{prefix}_mask.pth"))
            torch.save(dinov2_feats[i].cpu(), feat_filename)
            torch.save(masks[i].squeeze(0).cpu(), mask_filename)

print("Feature extraction complete!")

