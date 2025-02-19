
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# set the random seed
np.random.seed(0)

def save_imgs(imgs, path):
  for i, img in enumerate(imgs):
    # save as 00000.jpg, 00001.jpg, ...
    img.save(f"{path}/{i:05d}.jpg")

def show_points(coords, labels, ax, marker_size=200, color='green'):
    unknown_points = coords[labels==-1]
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(unknown_points[:, 0], unknown_points[:, 1], color=color, marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

data_root = f'/media/gvl/ACDA-BDB0/datasets/bridge/bridge_dataset/1.0.0'
b_tfds = tfds.builder_from_directory(builder_dir=data_root)

samples_all = {
    'train': {},
    'val': {}
}

world2pixel = np.load('project_matrix.npy')

for trainval in ['train', 'val']:
    ds = b_tfds.as_dataset(split=trainval, shuffle_files=True)
    for epi_ind, episode in tqdm(enumerate(ds)):
        img_save_dir = f'demo/{trainval}/{epi_ind}'
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        images = [step['observation']['image_0'] for step in episode['steps']]
        new_width = 256
        new_height = 256
        resized_images = []

        for image in images:
            resized_image = Image.fromarray(image.numpy()).resize((new_width, new_height))
            resized_images.append(resized_image)

        save_imgs(resized_images, img_save_dir)

        ee_traj_6d = [step['observation']['state'] for step in episode['steps']]
        ee_traj_6d = np.array(ee_traj_6d)
        ee_traj_3d = np.concatenate([ee_traj_6d[..., :3], np.ones_like(ee_traj_6d[:, :1])], axis=-1)


        reproj_2d = world2pixel @ ee_traj_3d.T
        reproj_2d = reproj_2d[:2] / reproj_2d[2]
        reproj_2d = reproj_2d.T.astype(int)
        

        vis_dir = img_save_dir.replace(trainval, f'{trainval}_eevis')
        os.makedirs(vis_dir, exist_ok=True)
        output_video_path = f'{vis_dir}.mp4'

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(img_save_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        plt.close("all")
        frame_paths = []
        for out_frame_idx in range(0, len(frame_names)):
            if out_frame_idx % 1 == 0:
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")
                plt.imshow(Image.open(os.path.join(img_save_dir, frame_names[out_frame_idx])))
                show_points(np.array([reproj_2d[out_frame_idx]]), np.array([-1]), plt.gca(), color='red')

                frame_path = os.path.join(vis_dir, f"frame_{out_frame_idx:04d}.png")
                plt.axis('off')  # Remove axis for cleaner visualization
                plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                frame_paths.append(frame_path)

        # Create video from frames
        frame = cv2.imread(frame_paths[0])
        height, width, _ = frame.shape
        fps = 5  # Adjust frame rate as needed

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            video_writer.write(img)

        video_writer.release()

        # remove the frames
        for frame_path in frame_paths:
            os.remove(frame_path)

        if epi_ind == 10:
            break