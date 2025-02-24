# import tensorflow_datasets as tfds
# import tensorflow as tf
# import dlimp as dl
# from tqdm import tqdm
import os
import jsonlines

# for gpu_id in range(3, 4):
#     trainval = "train"
#     data_path = '/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/OXE'
#     total_size = 53192
#     chunk_size = total_size // 8
#     start_idx = gpu_id * chunk_size
#     end_idx = start_idx + chunk_size
#     trainval = f"{trainval}[{start_idx}:{end_idx}]"

#     work_list = []

#     with tf.device('/CPU:0'):
#         builder = tfds.builder("bridge_orig", data_dir=data_path)
#         dataset = dl.DLataset.from_rlds(builder, split=trainval, shuffle=False, num_parallel_reads=tf.data.AUTOTUNE)
#         dataset = dataset.apply(tf.data.experimental.ignore_errors())
#         for i, sample in tqdm(enumerate(dataset.as_numpy_iterator())):
#             episode_file = sample['traj_metadata']['episode_metadata']['file_path'][0].decode('utf-8')
#             episode_id = sample["traj_metadata"]["episode_metadata"]["episode_id"][0]
#             unique_id = f"{episode_file}~{episode_id}"
#             work_list.append(unique_id)

#             with open(f"gpu_{gpu_id}.txt", "a") as f:
#                 f.write(f"{unique_id}\n")

import json

def get_unique_dicts(jsonl_file):
    unique_dicts = set()
    
    with jsonlines.open(jsonl_file, "r") as f:
        for episode in f:
            unique_dicts.add(json.dumps(episode, sort_keys=True))
    
    return [json.loads(item) for item in unique_dicts]  # Convert back to dicts

# Example usage
if __name__ == "__main__":
    jsonl_list = os.listdir("data_preprocessing/meta_data/val/")
    jsonl_list = [f"data_preprocessing/meta_data/val/{f}" for f in jsonl_list if f.endswith(".jsonl")]
    processed_episodes = []

    for jsonl_file in jsonl_list:
        unique_dicts = get_unique_dicts(jsonl_file)
        processed_episodes.extend(unique_dicts)
    unique_processed_episodes = set()
    for episode in processed_episodes:
        unique_processed_episodes.add(json.dumps(episode, sort_keys=True))
    
    unique_processed_episodes = [json.loads(item) for item in unique_processed_episodes]
    print(f"Unique processed episodes: {len(unique_processed_episodes)}")
    print(f"adding {len(unique_processed_episodes)} to progress_dict.json")

    progress_dict = {}
    if os.path.exists("data_preprocessing/meta_data/val/progress_dict.json"):
        with open("data_preprocessing/meta_data/val/progress_dict.json", "r") as f:
            progress_dict = json.load(f)
    print(f"Current progress_dict length: {len(progress_dict)}")

    for episode in unique_processed_episodes:
        key, value = next(iter(episode.items()))
        if key not in progress_dict:
            progress_dict[key] = value

    with open("data_preprocessing/meta_data/val/progress_dict.json", "w") as f:
        json.dump(progress_dict, f, indent=4)

    print(f"Updated progress_dict length: {len(progress_dict)}")

    # for gpu_id in range(8):
    #     task_list = []
    #     with open(f"data_preprocessing/meta_data/train/tmp/gpu_{gpu_id}.txt", "r") as f:
    #         for line in f:
    #             task_list.append(line.strip())

    #     modified_jsonl = f"data_preprocessing/meta_data/train/estimated_poses{gpu_id}.jsonl"
    #     for episode in unique_processed_episodes:
    #         key, value = next(iter(episode.items()))
    #         if key in task_list:
    #             with jsonlines.open(modified_jsonl, "a") as f:
    #                 f.write({
    #                     key: value
    #                 })
            