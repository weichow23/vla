## debug
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
from PIL import Image
import torch
from data_preprocessing.ee_tracks_extractionv2 import get_args_parser, init_DINO_model
import numpy as np
from torchvision import transforms

parser = get_args_parser()
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img_dir = "data/images/"
mask_dir = "data/images/masks_all/"
feature_save_path = "data/features/"
os.makedirs(feature_save_path, exist_ok=True)

## TODO Just for testing
model, base_transform, denormalizer = init_DINO_model(
    model_identifier=args.DINO_model,
    stride=args.stride,
    img_size=(args.img_size, args.img_size),
    patch_size=args.patch_size,
    device=device,
) 

dinov2_feats, labels = [], []
frames = os.listdir(img_dir)
feat_size = 37
# frames = [f for f in frames if f.endswith(".jpg")]
# frames = sorted(frames)
frames = [f'img{i}.jpg' for i in range(0, 10)]
for scene in tqdm(frames):
    frame_path = os.path.join(img_dir, scene)
    img = Image.open(frame_path).convert("RGB")
    img = base_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        dinov2_feat = model.forward_features(img)[:, 1:]
        dinov2_feat = dinov2_feat.permute(0, 2, 1).reshape(-1, 768, 37, 37)
    annotation_pth = f"{mask_dir}/{scene.replace('.jpg', '.png')}"
    annotation = Image.open(annotation_pth)
    annotation = np.array(annotation)
    annotation = annotation[..., 3] 
    annotation = Image.fromarray(annotation)
    annotation = transforms.Resize((feat_size, feat_size), Image.NEAREST)(annotation)
    annotation = transforms.ToTensor()(annotation).squeeze(0)
    annotation = (annotation > 0.).float().cuda()
    dinov2_feats.append(dinov2_feat.cuda())
    labels.append(annotation)

dinov2_feats = torch.concatenate(dinov2_feats).permute(0, 2, 3, 1).contiguous()
labels = torch.stack(labels)
labels = labels * 2 - 1

breakpoint()

subsampled_flattened_dinov2_feats = dinov2_feats.view(-1, 768) # [::4]
subsampled_labels = labels.view(-1, 1) # [::4]
X_pinv_dinov2 = torch.pinverse(subsampled_flattened_dinov2_feats)
to_foreground_dinov2 = torch.matmul(X_pinv_dinov2, subsampled_labels)
print(to_foreground_dinov2.shape)

# test_frames = os.listdir("data/images")
# test_frames = [f for f in test_frames if f.endswith(".jpg")]
# test_frames = sorted(test_frames)[:10]
test_frames = frames

fig, axes = plt.subplots(len(test_frames), 2, figsize=(10, 50))
for i in range(len(test_frames)):
    img = Image.open(f"data/images/{test_frames[i]}").convert("RGB")
    img = base_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # dinov2_feat = model.forward_intermediates(
        #     img,
        #     norm=True,
        #     output_fmt="NCHW",
        #     intermediates_only=True,
        # )[-1].permute(0, 2, 3, 1)
        dinov2_feat = model.forward_features(img)[:, 1:]
        dinov2_feat = dinov2_feat.permute(0, 2, 1).reshape(-1, 768, 37, 37)

    image = imageio.imread(f"data/images/{test_frames[i]}")
    image_shape = image.shape
    
    dinov2_feat = dinov2_feat.permute(0, 2, 3, 1)
    predicted_dinov2 = dinov2_feat @ to_foreground_dinov2
    predicted_dinov2 = predicted_dinov2.reshape(feat_size, feat_size).cpu().numpy()
    predicted_dinov2 = np.where(predicted_dinov2 < 0, 0, predicted_dinov2)
    predicted_dinov2 = (predicted_dinov2 - predicted_dinov2.min()) / (predicted_dinov2.max() - predicted_dinov2.min())
    predicted_dinov2 = (predicted_dinov2 * 255).astype(np.uint8)
    predicted_dinov2 = Image.fromarray(predicted_dinov2).resize((image_shape[1], image_shape[0]))
    
    axes[i, 0].imshow(image)
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(predicted_dinov2, cmap='gray')
    axes[i, 1].set_title('Predicted Foreground (dinov2)')
    axes[i, 1].axis('off')
plt.savefig("small_size_foreground_dinov2.png", bbox_inches='tight', dpi=300)
plt.close(fig)