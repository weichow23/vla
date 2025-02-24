import h5py
import os

def merge_hdf5(input_files, output_file):
    with h5py.File(output_file, 'w') as f:
        for i, file in enumerate(input_files):
            with h5py.File(file, 'r') as f_in:
                for key in f_in.keys():
                    if key in f:
                        print(f"Key {key} already exists in the output file. Skipping...")
                        continue
                    f.create_group(key)
                    for subkey in f_in[key].keys():
                        f[key].create_dataset(subkey, data=f_in[key][subkey])
            print(f"File {i+1}/{len(input_files)} processed.")
    print(f"Output file saved to {output_file}")

if __name__ == '__main__':
    data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/OXE/raw_cam_vla/"
    h5_files = os.listdir(data_path)
    h5_files = [os.path.join(data_path, f) for f in h5_files if f.endswith('.h5')]

    breakpoint()

    merge_hdf5(h5_files, f'{data_path}/merged.h5')