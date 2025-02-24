#!/bin/bash

for gpu_id in {0..7}
do
    # python data_preprocessing/ee_tracks_extractionv2_tfds.py --visualize --vis_path vis --trainval val --num_episodes 6872 --gpu_id $gpu_id  &
    python data_preprocessing/ee_tracks_extractionv2_tfds.py --visualize --vis_path vis_train --trainval train --num_episodes 53192 --gpu_id $gpu_id  & # 53192
done

wait  # Wait for all background processes to finish before exiting the script

