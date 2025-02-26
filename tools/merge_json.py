import json

def merge_json(file_list, output_file):
    all_data = {}
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.update(data)

    invalid = []
    for key in all_data.keys():
        if all_data[key]["0"] is None:
            invalid.append(key)

    invalid_val = []
    for key in invalid:
        if 'val' in key:
            invalid_val.append(key)

    # with open(output_file, 'w') as f:
    #     json.dump(all_data, f, indent=4)

if __name__ == '__main__':
    jsons = ["data_preprocessing/meta_data/train/progress_dict.json", "data_preprocessing/meta_data/val/progress_dict.json"]
    merge_json(jsons, "/lustre/fsw/portfolios/nvr/projects/nvr_av_foundations/STORRM/OXE/bridge_orig/camera_calibration.json")