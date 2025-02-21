"""
Precompute the DINO v2 features and train the gripper classifier.
"""
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF
from data_preprocessing.gripper_classifier import BNHead
from data_preprocessing.ee_tracks_extractionv2 import get_args_parser, init_DINO_model

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        img_files = os.listdir(img_dir)
        self.img_files = [f for f in img_files if f.endswith(".jpg")]
        self.augment = augment
        self.img_size = img_size
        
        # Define the image transform: Resize → ToTensor → Normalize
        self.image_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Define the mask transform: Resize (Nearest) → ToTensor
        # (no normalization for mask)
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

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
        mask = mask[..., 3] 
        mask = Image.fromarray(mask)

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

        # Apply transforms
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.).float()  # binarize mask

        assert mask.max() > 0.0, f"Mask max: {mask.max()}"

        return img, mask, img_file

class FeatureDataset(Dataset):
    def __init__(self, feature_dir, label_dir, img_size):
        self.feature_dir = feature_dir
        self.feature_files = os.listdir(feature_dir)
        self.feature_files = [f for f in self.feature_files if f.endswith(".pth")]
        
        self.label_dir = label_dir

        self.img_size = img_size

        # Define the mask transform: Resize (Nearest) → ToTensor
        # (no normalization for mask)
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        feature_path = os.path.join(self.feature_dir, feature_file)
        label_path = os.path.join(self.label_dir, feature_file[:-8] + ".png")

        feature = torch.load(feature_path, map_location="cpu").to(torch.float32)
        label = Image.open(label_path)
        label = np.array(label)
        label = label[..., 3] 
        label = Image.fromarray(label)
        label = self.mask_transform(label)
        label = (label > 0.).long()

        assert label.max() > 0.0, f"Label max: {label.max()}"

        return feature, label

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Configuration
    Image.MAX_IMAGE_PIXELS = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(110)
    torch.manual_seed(110)

    classifier = BNHead(num_channels=768, output_shape=(args.img_size, args.img_size)).to(device)
    # classifier.load_state_dict(torch.load(args.classifier_model.replace('.pth', '_v2.pth'))) # Load the pre-trained classifier

    ## prepare the features for training
    # Load dataset
    img_dir = "data/images/"
    mask_dir = "data/images/masks_all/"
    dataset = ImageDataset(img_dir, mask_dir, (args.img_size, args.img_size), augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    prefix = "000"
    num_epochs = 100

    # Feature extraction and saving
    feature_save_path = "data/features/"
    os.makedirs(feature_save_path, exist_ok=True)

    if len(os.listdir(feature_save_path)) != len(dataset):
        print(f"Extracting features since {len(os.listdir(feature_save_path))} features found in {feature_save_path} but {len(dataset)} expected.")
        # cache the features if not already done
        model, base_transform, denormalizer = init_DINO_model(
            model_identifier=args.DINO_model,
            stride=args.stride,
            img_size=(args.img_size, args.img_size),
            patch_size=args.patch_size,
            device=device,
        ) 

        model.eval()
        with torch.no_grad():
            for imgs, masks, img_files in tqdm(dataloader, desc="Processing Batches"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                # dinov2_feats = model.forward_intermediates(
                #     one_batch, 
                #     norm=True,
                #     output_fmt="NCHW",
                #     intermediates_only=True
                # )[-1].permute(0, 2, 3, 1)
                dinov2_feats = model.forward_features(imgs)[:, 1:]
                dinov2_feats = dinov2_feats.permute(0, 2, 1).reshape(-1, 768, 37, 37)
                
                for i in range(len(img_files)):
                    feat_filename = os.path.join(feature_save_path, img_files[i].replace(".jpg", f"_{prefix}.pth"))
                    torch.save(dinov2_feats[i].cpu(), feat_filename)
        print("Feature extraction complete!")

    # Now start training the classifier
    dataset = FeatureDataset(feature_save_path, mask_dir, (args.img_size, args.img_size))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=12, drop_last=True, pin_memory=True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    
    for epoch in tqdm(range(num_epochs)):
        classifier.train()
        epoch_loss = 0
        for features, labels in dataloader:
            features = features.squeeze().to(device)
            labels = labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = classifier(features.to(torch.float32)).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
    torch.save(classifier.state_dict(), f"{args.classifier_model.replace('.pth', '_v2.pth')}")

if __name__=="__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)