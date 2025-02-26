"""
This is not used. But maybe we should add ego_state to the dataset files at some point.
"""

import tensorflow as tf
import os
import glob

def compute_ego_state(episode):
    """
    Dummy function to compute ego_state.
    Replace this with the actual function that generates the ego_state.
    """
    # Example: Generate a zero tensor for ego_state
    ego_state = tf.zeros([10], dtype=tf.float32)  # Modify as per your need
    return ego_state

def modify_tfrecord(input_file, output_file):
    """
    Reads a TFRecord file, adds 'ego_state' to 'observation', and writes to a new file.
    """
    raw_dataset = tf.data.TFRecordDataset(input_file)
    writer = tf.io.TFRecordWriter(output_file)

    def modify_example(example_proto):
        # Define feature structure
        feature_description = {
            "observation": tf.io.FixedLenFeature([], tf.string),
            "traj_metadata": tf.io.FixedLenFeature([], tf.string)  # Adjust as needed
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)

        # Decode observation
        observation = tf.io.parse_tensor(parsed_example["observation"], out_type=tf.float32)

        # Compute new ego_state and add it to the observation
        ego_state = compute_ego_state(observation)  
        modified_observation = {
            "state": observation,  # Keep original state
            "ego_state": ego_state,  # Add new key
        }

        # Re-encode the modified observation
        parsed_example["observation"] = tf.io.serialize_tensor(modified_observation)

        # Serialize and write to new TFRecord
        modified_example = tf.train.Example(features=tf.train.Features(feature=parsed_example))
        writer.write(modified_example.SerializeToString())

    for raw_record in raw_dataset:
        modify_example(raw_record)

    writer.close()
    print(f"Modified {input_file} -> {output_file}")


if __name__ == "__main__":
    raw_tfds_files = glob.glob(os.path.join("/PATH/TO/OXE/bridge_orig/1.0.0", "bridge_dataset-train.tfrecord-*"))
    tgt_path = "/PATH/TO/OXE/bridge_mod/1.0.0"
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    raw_tfds_files.sort()
    breakpoint()
    for raw_tfds_file in raw_tfds_files:
        modify_tfrecord(raw_tfds_file, raw_tfds_file.replace("orig", "mod"))