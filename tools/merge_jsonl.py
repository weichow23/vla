import os
import jsonlines
import json

jsonls = os.listdir('data_preprocessing/meta_data/val')
jsonls = [f for f in jsonls if f.endswith('.jsonl')]

merged = []
seen = set()

for f in jsonls:
    with jsonlines.open(f'data_preprocessing/meta_data/val/{f}') as reader:
        for obj in reader:
            obj_str = json.dumps(obj, sort_keys=True)  # Convert dict to sorted string for uniqueness check
            if obj_str not in seen:
                seen.add(obj_str)
                merged.append(obj)

with jsonlines.open('data_preprocessing/meta_data/val/sample_all_val.jsonl', 'w') as writer:
    for obj in merged:
        writer.write(obj)