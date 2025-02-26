# import tensorflow_datasets as tfds
# import tensorflow as tf
# import dlimp as dl
# from tqdm import tqdm
import os
import jsonlines

import json

def get_unique_dicts(jsonl_file):
    unique_dicts = set()
    
    with jsonlines.open(jsonl_file, "r") as f:
        for episode in f:
            unique_dicts.add(json.dumps(episode, sort_keys=True))
    
    return [json.loads(item) for item in unique_dicts]  # Convert back to dicts

# Example usage
if __name__ == "__main__":
    trainval = "train"
    jsonl_list = os.listdir(f"data_preprocessing/meta_data/{trainval}/")
    jsonl_list = [f"data_preprocessing/meta_data/{trainval}/{f}" for f in jsonl_list if f.endswith(".jsonl")]
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
    if os.path.exists(f"data_preprocessing/meta_data/{trainval}/progress_dict.json"):
        with open(f"data_preprocessing/meta_data/{trainval}/progress_dict.json", "r") as f:
            progress_dict = json.load(f)
    print(f"Current progress_dict length: {len(progress_dict)}")

    for episode in unique_processed_episodes:
        key, value = next(iter(episode.items()))
        if key not in progress_dict:
            progress_dict[key] = value

    with open(f"data_preprocessing/meta_data/{trainval}/progress_dict.json", "w") as f:
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
            