import h5py
import os
def print_hdf5_structure(name, obj):
    print(name)

def find_all_groups_0(h5_obj, path="", results=None):
    """Recursively search for all groups named '0' in an HDF5 file."""
    if results is None:
        results = []

    for key in h5_obj.keys():
        current_path = f"{path}/{key}" if path else key
        if "~" in key:
            # print(f"Found group '0' at: {current_path}")
            results.append(current_path)

        if isinstance(h5_obj[key], h5py.Group):
            find_all_groups_0(h5_obj[key], current_path, results)

    return results

def merge_hdf5(input_files, output_file):
    all_episodes = []
    with h5py.File(output_file, 'a') as f:
        for i, file in enumerate(input_files):
            with h5py.File(file, 'r') as f_in:
                # f_in.visititems(print_hdf5_structure)
                groups = find_all_groups_0(f_in)
                all_episodes.extend(groups)
                for g in groups:
                    save_path = g
                    if save_path in f:
                        # print(f"Key {save_path} already exists in the output file. Skipping...")
                        continue
                    print("saving", save_path)
                    f.create_group(save_path)
                    for subkey in f_in[g].keys():
                        data = f_in[g][subkey]
                        if isinstance(data, h5py.Dataset):
                            f[save_path].create_dataset(subkey, data=data)
                        elif isinstance(data, h5py.Group):
                            f[save_path].create_group(subkey)
                            for subsubkey in data.keys():
                                f[save_path][subkey].create_dataset(subsubkey, data=data[subsubkey])
                        else:
                            print(f"Unknown data type: {type(data)}")
            print(f"File {i+1}/{len(input_files)} processed.")
    print(f"Output file saved to {output_file}")
    all_episodes = list(set(all_episodes))
    with open("raw_data_in_h5.txt", "w") as f:
        for ep in all_episodes:
            f.write(ep + "\n")

if __name__ == '__main__':
    data_path = "/PATH/TO/OXE/bridge_orig/raw_cam_vla/"
    h5_files = os.listdir(data_path)
    h5_files = [os.path.join(data_path, f) for f in h5_files if f.endswith('.h5') and not f.endswith('merged.h5')]

    # with h5py.File(f'{data_path}/merged.h5', 'r') as f_in:
    #     breakpoint()
    #     f_in.visititems(print_hdf5_structure)

    merge_hdf5(h5_files, f'{data_path}/merged.h5')