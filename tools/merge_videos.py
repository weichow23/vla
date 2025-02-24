import os
import argparse
from data_preprocessing.utils import concat_videos_grid


parser = argparse.ArgumentParser()
parser.add_argument("--videos_dir", type=str, default="vis")
parser.add_argument("--merge_name", type=str, default="all_videos.mp4")

if __name__ == "__main__":
    args = parser.parse_args()
    videos = os.listdir(args.videos_dir)
    videos = [os.path.join(args.videos_dir, v) for v in videos]
    concat_videos_grid(videos, f"{args.videos_dir}/{args.merge_name}")
    os.system(f"rm -rf {args.videos_dir}")